use std::cmp::{max, min};
use std::fmt;
use std::thread;

/// Used to build a graph of nodes, which can be evaluated and checked for constraints
pub struct Builder {
    nodes: Vec<Vec<Node>>,         // nodes[i] is the list of nodes at depth i
    values: Vec<Vec<Option<u64>>>, // cache of calculated node values, same structure as nodes
    equalities: Vec<Equality>,     // list of equalities to check
}

/// A node in the graph, which can be an input, constant, addition or multiplication of
/// two other nodes, or a hint, which executes an arbitrary function on another node.
/// Nodes are constructed using the builder.
#[derive(Debug, Clone, Copy)]
pub struct Node {
    indices: NodeIndices,
    node_type: NodeType,
}

#[derive(Debug, Clone, Copy)]
enum NodeType {
    Input,
    Const,
    Add(NodeIndices, NodeIndices),
    Mul(NodeIndices, NodeIndices),
    Hint(NodeIndices, fn(u64) -> u64),
}

#[derive(Debug, Clone, Copy)]
struct NodeIndices(usize, usize); // (depth, index) of a node in the graph

impl fmt::Display for NodeIndices {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Depth: {}, Index: {}", self.0, self.1)
    }
}

// Stores the indices of two nodes in the graph, which are asserted to be equal
struct Equality(NodeIndices, NodeIndices);

#[derive(Debug, Clone)]
pub enum EvaluationError {
    MissingInput,
    ValueDoesNotExist(String),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvaluationError::MissingInput => write!(
                f,
                "All inputs must be filled with set_input before evaluate is called"
            ),
            EvaluationError::ValueDoesNotExist(s) => write!(
                f,
                "Encountered None value in graph: {}, evauluate may not have been called",
                s
            ),
        }
    }
}

impl Builder {
    pub fn new() -> Self {
        let mut b = Builder {
            nodes: Vec::new(),
            values: Vec::new(),
            equalities: Vec::new(),
        };

        // Initialize the first layer of the graph, used for inputs and constants
        b.nodes.push(Vec::new());
        b.values.push(Vec::new());

        b
    }

    fn push_node(&mut self, nt: NodeType) -> &Node {
        // Calculate the depth of the new node: 0 for input/const, 1 more than deepest parent node for others
        let depth = match nt {
            NodeType::Input | NodeType::Const => 0,
            NodeType::Add(a, b) | NodeType::Mul(a, b) => max(a.0, b.0) + 1,
            NodeType::Hint(a, _) => a.0 + 1,
        };

        // Add new layers if necessary
        if depth >= self.nodes.len() {
            self.nodes.push(Vec::new());
            self.values.push(Vec::new());
        }

        // Push the new node onto the layer at calculated depth
        let layer = &mut self.nodes[depth];
        layer.push(Node {
            indices: NodeIndices(depth, layer.len()),
            node_type: nt,
        });

        // Initialize corresponding value cache
        self.values[depth].push(None);

        layer.last().unwrap() // safe to unwrap because we just pushed it
    }

    /// Returns the value of a node in the graph if it has been evaluated, or None if it has not
    pub fn get_value(&self, node: Node) -> Option<u64> {
        self.values[node.indices.0][node.indices.1]
    }

    // Caches the value of a node in the graph
    fn cache_node_value(&mut self, indices: NodeIndices, value: u64) {
        self.values[indices.0][indices.1] = Some(value);
    }

    // Helper function for evaluate(), calculates the value of a node in the graph
    fn calculate_value_helper(&self, node: &Node) -> Option<u64> {
        match &node.node_type {
            NodeType::Input => self.values[node.indices.0][node.indices.1],
            NodeType::Const => self.values[node.indices.0][node.indices.1],
            NodeType::Add(a, b) => {
                let a_val = self.values[a.0][a.1]?;
                let b_val = self.values[b.0][b.1]?;
                Some(a_val + b_val)
            }
            NodeType::Mul(a, b) => {
                let a_val = self.values[a.0][a.1]?;
                let b_val = self.values[b.0][b.1]?;
                Some(a_val * b_val)
            }
            NodeType::Hint(a, f) => {
                let a_val = self.values[a.0][a.1]?;
                Some(f(a_val))
            }
        }
    }

    /// Evaluates the graph, filling in the values of all nodes
    /// Returns an error if any input nodes are missing a value
    /// Multithreads the evaluation of each layer of the graph if possible, 
    /// defining "layer" as the set of nodes that can be evaluated with only the values from the previous layers
    pub fn evaluate(&mut self) -> Result<(), EvaluationError> {
        // Check that all input nodes have been filled
        for &node in &self.values[0] {
            if node == None {
                return Err(EvaluationError::MissingInput);
            }
        }

        // Get the number of threads available, defaulting to 1 if it fails
        let available_threads = match thread::available_parallelism() {
            Ok(s) => s.get(),
            Err(_) => {
                eprintln!("Failed to get available parallelism! Defaulting to singlethreaded");
                1
            }
        };

        // Iterate through each layer of the graph
        for (i, layer) in self.nodes[1..].iter().enumerate() {
            let depth = i + 1;
            // Use no more threads than nodes in the layer
            let used_threads = min(layer.len(), available_threads);

            // Singlethreaded evaluation
            if used_threads == 1 {
                let values = layer
                    .iter()
                    .map(|node| self.calculate_value_helper(node))
                    .collect();
                self.values[depth] = values;
                continue;
            }

            // Multithreaded evaluation
            // Split the layer into equally sized chunks, saving the remainder for the main thread
            let equal_chunks = layer.chunks_exact(layer.len() / (used_threads - 1));
            let remainder_chunk = equal_chunks.remainder();

            self.values[depth] = thread::scope(|s| -> Vec<Option<u64>> {
                let mut handles = vec![];
                let mut output = vec![];

                for chunk in equal_chunks {
                    // Spawn a thread to calculate the values of the nodes in the chunk, and collect it into a vector
                    handles.push(s.spawn(|| -> Vec<Option<u64>> {
                        chunk
                            .iter()
                            .map(|node| self.calculate_value_helper(node))
                            .collect()
                    }));
                }
                // Calculate the values of the remainder chunk in the main thread
                let mut remainder_vec: Vec<Option<u64>> = remainder_chunk
                    .iter()
                    .map(|node| self.calculate_value_helper(node))
                    .collect();

                // Collect the values from the threads and the remainder into a single vector and set the layer's values to it
                for handle in handles {
                    let mut chunk_vec = handle
                        .join()
                        .expect("Child thread panicked during evaulation!");
                    output.append(&mut chunk_vec);
                }
                output.append(&mut remainder_vec);
                output
            });
        }
        Ok(())
    }

    /// Initializes a node in the graph
    pub fn input(&mut self) -> Node {
        self.push_node(NodeType::Input).clone()
    }

    /// Initializes a node in a graph, set to a constant value
    pub fn constant(&mut self, value: u64) -> Node {
        let indices = self.push_node(NodeType::Const).indices;
        self.cache_node_value(indices, value);
        self.nodes[indices.0][indices.1].clone()
    }

    /// Adds 2 nodes in the graph, returning a new node
    pub fn add(&mut self, a: Node, b: Node) -> Node {
        self.push_node(NodeType::Add(a.indices, b.indices)).clone()
    }

    /// Multiplies 2 nodes in the graph, returning a new node
    pub fn mul(&mut self, a: Node, b: Node) -> Node {
        self.push_node(NodeType::Mul(a.indices, b.indices)).clone()
    }

    /// Adds a hint to the graph, given an input node and a function to apply to it, returning a new node
    pub fn hint(&mut self, a: Node, func: fn(u64) -> u64) -> Node {
        self.push_node(NodeType::Hint(a.indices, func)).clone()
    }

    /// Asserts that 2 nodes are equal
    pub fn assert_equal(&mut self, a: Node, b: Node) {
        self.equalities.push(Equality(a.indices, b.indices));
    }

    /// Sets the value of an input node, Panics if the node is not an input
    pub fn set_input(&mut self, node: Node, value: u64) {
        match node.node_type {
            NodeType::Input => self.cache_node_value(node.indices, value),
            _ => panic!("Tried to set input of non-input node!"),
        }
    }

    /// Given a graph that has `fill_nodes` already called on it, checks that all the constraints hold
    pub fn check_constraints(&mut self) -> Result<bool, EvaluationError> {
        for (i, eq) in self.equalities.iter().enumerate() {
            let left = self.values[eq.0 .0][eq.0 .1].ok_or(EvaluationError::ValueDoesNotExist(
                format!("Left hand side of equality #{}", i),
            ))?;
            let right = self.values[eq.1 .0][eq.1 .1].ok_or(EvaluationError::ValueDoesNotExist(
                format!("Right hand side of equality #{}", i),
            ))?;
            if left != right {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // https://hackmd.io/_uploads/BJFNjqwk6.png
    fn a_sq_plus_a_plus_5() {
        let mut builder = Builder::new();
        let a = builder.input();
        let a_squared = builder.mul(a, a);
        let five = builder.constant(5);
        let a_squared_plus_5 = builder.add(a_squared, five);
        let y = builder.add(a_squared_plus_5, a);

        let a_val = 3;
        builder.set_input(a, a_val);
        let expected = builder.constant(a_val * a_val + a_val + 5);
        builder.assert_equal(y, expected);

        assert_eq!(builder.evaluate().unwrap(), ());
        assert!(builder.check_constraints().unwrap());
    }

    #[test]
    // https://hackmd.io/_uploads/ryo4acvyp.png
    fn a_plus_1_all_over_8() {
        let mut builder = Builder::new();
        let a = builder.input();
        let one = builder.constant(1);
        let b = builder.add(a, one);
        let c = builder.hint(b, |x: u64| x / 8);
        let eight = builder.constant(8);
        let c_times_8 = builder.mul(c, eight);

        builder.assert_equal(b, c_times_8);

        // Note: A+1 must be a multiple of 8 for this to work under integer division
        builder.set_input(a, 127);

        assert_eq!(builder.evaluate().unwrap(), ());
        assert!(builder.check_constraints().unwrap());
    }

    #[test]
    fn multithreading() {
        // Layer 0
        let mut builder = Builder::new();
        let a = builder.input();
        let b = builder.input();
        let c = builder.input();
        builder.set_input(a, 1);
        builder.set_input(b, 2);
        builder.set_input(c, 3);
        let d = builder.constant(4);
        let e = builder.constant(5);
        let f = builder.constant(6);
        let g = builder.constant(7);
        let h = builder.constant(8);

        // Layer 1
        let ab = builder.mul(a, b);
        let cd = builder.mul(c, d);
        let ef = builder.mul(e, f);
        let gh = builder.mul(g, h);
        // Spam a bunch of nodes to give the threads something to do
        for _ in 0..100 {
            let _ag = builder.mul(a, g);
        }

        // Layer 2
        let abc = builder.mul(ab, c); // Depth 1 * Depth 0
        let cdef = builder.mul(cd, ef); // Depth 1 * Depth 1
        for _ in 0..100 {
            let _abgh = builder.mul(ab, gh);
        }

        // Layer 3
        let abcgh = builder.mul(abc, gh); // Depth 2 * Depth 1
        for _ in 0..100 {
            let _abcef = builder.mul(abc, ef);
        }
        // Add a node after spam to make sure multithreaded vectors are combined in the correct order
        let cdefgh = builder.mul(cdef, gh); // Depth 2 * Depth 1

        // Layer 4
        let abcghcdef = builder.mul(abcgh, cdef); // Depth 3 * Depth 2

        assert_eq!(builder.evaluate().unwrap(), ());

        // Make sure every single value in the graph is evaluated
        for i in 0..5 {
            let nodes_layer = &builder.nodes[i];
            let values_layer = &builder.values[i];
            assert!(nodes_layer.len() == values_layer.len());
            for val in values_layer {
                assert!(val.is_some());
            }
        }

        assert_eq!(builder.get_value(ab).unwrap(), 1 * 2);
        assert_eq!(builder.get_value(abc).unwrap(), 1 * 2 * 3);
        assert_eq!(builder.get_value(abcgh).unwrap(), 1 * 2 * 3 * 7 * 8);
        assert_eq!(builder.get_value(cdefgh).unwrap(), 3 * 4 * 5 * 6 * 7 * 8);
        assert_eq!(builder.get_value(cdefgh).unwrap(), 3 * 4 * 5 * 6 * 7 * 8);
        assert_eq!(
            builder.get_value(abcghcdef).unwrap(),
            1 * 2 * 3 * 7 * 8 * 3 * 4 * 5 * 6
        );
    }

    #[test]
    fn constraints_dont_hold() {
        let mut builder = Builder::new();
        let a = builder.input();
        let b = builder.constant(8);
        let c = builder.mul(a, b);
        let d = builder.constant(45);
        let e = builder.constant(46);

        builder.set_input(a, 6);

        builder.assert_equal(c, d);
        builder.assert_equal(c, e);

        assert_eq!(builder.evaluate().unwrap(), ());
        assert!(!builder.check_constraints().unwrap());
    }

    #[test]
    fn constraints_not_evaluated() {
        let mut builder = Builder::new();
        let a = builder.input();
        let b = builder.constant(8);
        let c = builder.mul(a, b);
        let d = builder.constant(45);

        builder.set_input(a, 6);

        builder.assert_equal(c, d);

        if let Err(EvaluationError::ValueDoesNotExist(_)) = builder.check_constraints() {
        } else {
            panic!("Expected EvaluationError::ValueDoesNotExist, got something else");
        }
    }

    #[test]
    fn empty_graph() {
        let mut builder = Builder::new();
        assert_eq!(builder.evaluate().unwrap(), ());
        assert!(builder.check_constraints().unwrap());
    }

    #[test]
    fn missing_input() {
        let mut builder = Builder::new();
        let a = builder.input();
        let b = builder.constant(8);
        let c = builder.mul(a, b);
        let d = builder.constant(45);

        builder.assert_equal(c, d);

        if let Err(EvaluationError::MissingInput) = builder.evaluate() {
        } else {
            panic!("Expected EvaluationError::MissingInput, got something else");
        }
    }

    #[test]
    fn set_input() {
        let mut builder = Builder::new();
        let x = builder.input();
        builder.set_input(x, 5);

        assert_eq!(builder.get_value(x).unwrap(), 5);
    }

    #[test]
    #[should_panic(expected = "Tried to set input of non-input node!")]
    fn set_input_on_non_input() {
        let mut builder = Builder::new();
        let x = builder.constant(0);
        builder.set_input(x, 123);
    }

    #[test]
    fn constant() {
        let mut builder = Builder::new();
        let x = builder.constant(1337);

        assert_eq!(builder.get_value(x).unwrap(), 1337);
    }

    #[test]
    fn add() {
        let mut builder = Builder::new();
        let a = builder.constant(2);
        let b = builder.constant(3);
        let result = builder.add(a, b);

        assert_eq!(builder.evaluate().unwrap(), ());
        assert_eq!(builder.get_value(result).unwrap(), 5);
    }

    #[test]
    fn mul() {
        let mut builder = Builder::new();
        let a = builder.constant(2);
        let b = builder.constant(3);
        let result = builder.mul(a, b);

        assert_eq!(builder.evaluate().unwrap(), ());
        assert_eq!(builder.get_value(result).unwrap(), 6);
    }

    #[test]
    fn hint() {
        let mut builder = Builder::new();
        let a = builder.constant(12);
        let result = builder.hint(a, |x: u64| x / 3);

        assert_eq!(builder.evaluate().unwrap(), ());
        assert_eq!(builder.get_value(result).unwrap(), 4);
    }
}
