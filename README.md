# inpoly

Fast point-in-polygon testing using the crossing-number algorithm.

Efficiently determines whether points lie inside, outside, or on the boundary of polygons (including those with holes).

## Usage

```rust
use ndarray::array;
use inpoly::inpoly2;

// Define a square polygon
let node = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

// Test some points
let points = array![
    [0.5, 0.5],  // inside
    [2.0, 2.0],  // outside
    [0.5, 0.0],  // on edge
];

let (inside, on_boundary) = inpoly2(&points, &node, None, None);

assert!(inside[0]);   // point is inside
assert!(!inside[1]);  // point is outside
assert!(inside[2]);   // point on edge counts as inside
assert!(on_boundary[2]); // detected as on boundary
```

## Polygons with Holes

```rust
use ndarray::array;
use inpoly::inpoly2;

// Diamond with a square hole
let node = array![
    // Outer diamond
    [4.0, 0.0], [8.0, 4.0], [4.0, 8.0], [0.0, 4.0],
    // Inner square (hole)
    [3.0, 3.0], [5.0, 3.0], [5.0, 5.0], [3.0, 5.0]
];

let edge = array![
    // Outer diamond edges
    [0, 1], [1, 2], [2, 3], [3, 0],
    // Inner hole edges
    [4, 5], [5, 6], [6, 7], [7, 4]
];

let points = array![[4.0, 4.0], [6.0, 4.0]];
let (inside, _) = inpoly2(&points, &node, Some(&edge), None);

assert!(!inside[0]); // center of hole is outside
assert!(inside[1]);  // between hole and outer edge is inside
```

## License

MIT
