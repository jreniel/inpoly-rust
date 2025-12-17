//! Fast point-in-polygon testing using the crossing-number algorithm.
//!
//! Efficiently determines whether points lie inside, outside, or on the
//! boundary of polygons (including those with holes).

use ndarray::{Array1, Array2, Axis};
use ndarray_stats::QuantileExt;

/// Find insertion points for values in a sorted array (left side).
fn searchsorted_left(sorted: &[f64], values: &[f64]) -> Vec<usize> {
    values
        .iter()
        .map(|&v| {
            sorted
                .binary_search_by(|&x| x.partial_cmp(&v).unwrap())
                .unwrap_or_else(|i| i)
        })
        .collect()
}

/// Find insertion points for values in a sorted array (right side).
fn searchsorted_right(sorted: &[f64], values: &[f64]) -> Vec<usize> {
    values
        .iter()
        .map(|&v| {
            match sorted.binary_search_by(|&x| x.partial_cmp(&v).unwrap()) {
                Ok(mut pos) => {
                    while pos < sorted.len() && sorted[pos] == v {
                        pos += 1;
                    }
                    pos
                }
                Err(pos) => pos,
            }
        })
        .collect()
}

/// Crossing-number test for point-in-polygon detection.
///
/// Loops over edges, uses binary-search to find vertices that intersect
/// each edge's y-range, then performs crossing-number comparisons.
fn crossing_number_test(
    vert: &Array2<f64>,
    node: &Array2<f64>,
    edge: &Array2<usize>,
    ftol: f64,
    lbar: f64,
) -> (Array1<bool>, Array1<bool>) {
    let feps = ftol * lbar;
    let veps = ftol * lbar;

    let vnum = vert.nrows();
    let mut stat = Array1::from_elem(vnum, false);
    let mut bnds = Array1::from_elem(vnum, false);

    // Extract edge endpoint coordinates
    let edge_col0: Vec<usize> = edge.column(0).to_vec();
    let edge_col1: Vec<usize> = edge.column(1).to_vec();

    let xone: Vec<f64> = edge_col0.iter().map(|&i| node[[i, 0]]).collect();
    let xtwo: Vec<f64> = edge_col1.iter().map(|&i| node[[i, 0]]).collect();
    let yone: Vec<f64> = edge_col0.iter().map(|&i| node[[i, 1]]).collect();
    let ytwo: Vec<f64> = edge_col1.iter().map(|&i| node[[i, 1]]).collect();

    // Compute edge bounding boxes
    let xmin: Vec<f64> = xone
        .iter()
        .zip(&xtwo)
        .map(|(&a, &b)| a.min(b) - veps)
        .collect();
    let xmax: Vec<f64> = xone
        .iter()
        .zip(&xtwo)
        .map(|(&a, &b)| a.max(b) + veps)
        .collect();
    let ymin: Vec<f64> = yone.iter().map(|&y| y - veps).collect();
    let ymax: Vec<f64> = ytwo.iter().map(|&y| y + veps).collect();

    // Edge deltas
    let ydel: Vec<f64> = ytwo.iter().zip(&yone).map(|(&a, &b)| a - b).collect();
    let xdel: Vec<f64> = xtwo.iter().zip(&xone).map(|(&a, &b)| a - b).collect();
    let edel: Vec<f64> = xdel
        .iter()
        .zip(&ydel)
        .map(|(&x, &y)| x.abs() + y)
        .collect();

    // Get sorted indices based on y-values
    let y_column: Vec<f64> = vert.column(1).to_vec();
    let mut ivec: Vec<usize> = (0..vnum).collect();
    ivec.sort_by(|&a, &b| y_column[a].partial_cmp(&y_column[b]).unwrap());

    // Create sorted y-values for binary search
    let y_sorted: Vec<f64> = ivec.iter().map(|&i| y_column[i]).collect();

    // Find y-range overlaps using binary search on SORTED values
    let ione = searchsorted_left(&y_sorted, &ymin);
    let itwo = searchsorted_right(&y_sorted, &ymax);

    // Loop over polygon edges
    for epos in 0..edge.nrows() {
        let e_xone = xone[epos];
        let e_xtwo = xtwo[epos];
        let e_yone = yone[epos];
        let e_ytwo = ytwo[epos];
        let e_xmin = xmin[epos];
        let e_xmax = xmax[epos];
        let e_edel = edel[epos];
        let e_xdel = xdel[epos];
        let e_ydel = ydel[epos];

        for &jvrt in ivec.iter().take(itwo[epos]).skip(ione[epos]) {
            if bnds[jvrt] {
                continue;
            }

            let xpos = vert[[jvrt, 0]];
            let ypos = vert[[jvrt, 1]];

            if xpos >= e_xmin {
                if xpos <= e_xmax {
                    // Compute crossing number
                    let mul1 = e_ydel * (xpos - e_xone);
                    let mul2 = e_xdel * (ypos - e_yone);

                    if feps * e_edel >= (mul2 - mul1).abs() {
                        // BNDS -- approximately on edge
                        bnds[jvrt] = true;
                        stat[jvrt] = true;
                    } else if ypos == e_yone && xpos == e_xone {
                        // BNDS -- match endpoint ONE
                        bnds[jvrt] = true;
                        stat[jvrt] = true;
                    } else if ypos == e_ytwo && xpos == e_xtwo {
                        // BNDS -- match endpoint TWO
                        bnds[jvrt] = true;
                        stat[jvrt] = true;
                    } else if mul1 <= mul2 && ypos >= e_yone && ypos < e_ytwo {
                        // Advance crossing number
                        stat[jvrt] = !stat[jvrt];
                    }
                }
            } else if ypos >= e_yone && ypos < e_ytwo {
                // Advance crossing number
                stat[jvrt] = !stat[jvrt];
            }
        }
    }

    (stat, bnds)
}

/// Test if points are inside a polygon.
///
/// Uses the crossing-number algorithm to determine if each vertex is inside,
/// outside, or on the boundary of a polygon defined by nodes and edges.
///
/// # Arguments
///
/// * `vert` - Query points as Nx2 array of (x, y) coordinates
/// * `node` - Polygon vertices as Mx2 array of (x, y) coordinates
/// * `edge` - Polygon edges as Kx2 array of node indices. If `None`, assumes
///   nodes form a closed loop (0→1→2→...→n-1→0)
/// * `ftol` - Floating-point tolerance for boundary detection. Default: 5e-14
///
/// # Returns
///
/// A tuple of two boolean arrays:
/// * `stat` - True if point is inside or on boundary
/// * `bnds` - True if point is on boundary
///
/// # Example
///
/// ```
/// use ndarray::array;
/// use inpoly::inpoly2;
///
/// let node = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let point = array![[0.5, 0.5]];
/// let (inside, on_boundary) = inpoly2(&point, &node, None, None);
/// assert!(inside[0]);  // point is inside the square
/// ```
pub fn inpoly2(
    vert: &Array2<f64>,
    node: &Array2<f64>,
    edge: Option<&Array2<usize>>,
    ftol: Option<f64>,
) -> (Array1<bool>, Array1<bool>) {
    let vnum = vert.nrows();
    let mut stat = Array1::from_elem(vnum, false);
    let mut bnds = Array1::from_elem(vnum, false);

    if node.is_empty() {
        return (stat, bnds);
    }

    // Build default edge connectivity if not provided
    let default_edge;
    let edge = match edge {
        Some(e) => e,
        None => {
            let n = node.nrows();
            let mut e = Array2::zeros((n, 2));
            for i in 0..n - 1 {
                e[[i, 0]] = i;
                e[[i, 1]] = i + 1;
            }
            e[[n - 1, 0]] = n - 1;
            e[[n - 1, 1]] = 0;
            default_edge = e;
            &default_edge
        }
    };

    // Compute bounding box of polygon
    let used: Vec<usize> = edge.iter().copied().collect();
    let sel = node.select(Axis(0), &used);
    let xmin = *sel.column(0).min().unwrap();
    let xmax = *sel.column(0).max().unwrap();
    let ymin = *sel.column(1).min().unwrap();
    let ymax = *sel.column(1).max().unwrap();

    let xdel = xmax - xmin;
    let ydel = ymax - ymin;
    let lbar = (xdel + ydel) / 2.0;
    let ftol = ftol.unwrap_or(5.0e-14);
    let feps = ftol * lbar;

    // Filter vertices to bounding box
    let mask: Vec<bool> = vert
        .outer_iter()
        .map(|row| {
            row[0] >= xmin - feps
                && row[0] <= xmax + feps
                && row[1] >= ymin - feps
                && row[1] <= ymax + feps
        })
        .collect();

    let selected_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();

    if selected_indices.is_empty() {
        return (stat, bnds);
    }

    let mut vert_subset = vert.select(Axis(0), &selected_indices);
    let mut node = node.clone();

    // Flip axes if x-range > y-range (ensures y is the "long" axis)
    if xdel > ydel {
        vert_subset.invert_axis(Axis(1));
        node.invert_axis(Axis(1));
    }

    // Sort edges so that yone <= ytwo
    let mut edge = edge.clone();
    for i in 0..edge.nrows() {
        let i0 = edge[[i, 0]];
        let i1 = edge[[i, 1]];
        if node[[i1, 1]] < node[[i0, 1]] {
            edge[[i, 0]] = i1;
            edge[[i, 1]] = i0;
        }
    }

    let (subset_stat, subset_bnds) = crossing_number_test(&vert_subset, &node, &edge, ftol, lbar);

    // Map results back to original indices
    for (i, &si) in selected_indices.iter().enumerate() {
        stat[si] = subset_stat[i];
        bnds[si] = subset_bnds[i];
    }

    (stat, bnds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn point_inside_square() {
        let node = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let point = array![[0.5, 0.5]];
        let (inpoly, onedge) = inpoly2(&point, &node, None, None);
        assert!(inpoly[0], "point should be inside");
        assert!(!onedge[0], "point should not be on edge");
    }

    #[test]
    fn point_outside_square() {
        let node = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let point = array![[2.0, 2.0]];
        let (inpoly, onedge) = inpoly2(&point, &node, None, None);
        assert!(!inpoly[0], "point should be outside");
        assert!(!onedge[0], "point should not be on edge");
    }

    #[test]
    fn point_on_edge() {
        let node = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let point = array![[0.5, 0.0]];
        let (inpoly, onedge) = inpoly2(&point, &node, None, None);
        assert!(inpoly[0], "point on edge should be considered inside");
        assert!(onedge[0], "point should be on edge");
    }

    #[test]
    fn point_on_vertex() {
        let node = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let point = array![[0.0, 0.0]];
        let (inpoly, onedge) = inpoly2(&point, &node, None, None);
        assert!(inpoly[0], "vertex point should be considered inside");
        assert!(onedge[0], "vertex should be on edge");
    }

    #[test]
    fn diamond_with_hole() {
        // Diamond with a square hole in the center
        let node = array![
            [4.0, 0.0],
            [8.0, 4.0],
            [4.0, 8.0],
            [0.0, 4.0],
            [3.0, 3.0],
            [5.0, 3.0],
            [5.0, 5.0],
            [3.0, 5.0]
        ];
        let edge = array![
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4]
        ];

        let points = array![
            [4.0, 4.0],  // center of hole - outside
            [6.0, 4.0],  // between hole and outer edge - inside
            [4.0, 1.0],  // inside diamond, below hole - inside
            [-1.0, 4.0], // outside diamond entirely
        ];

        let (inpoly, _onedge) = inpoly2(&points, &node, Some(&edge), None);

        assert!(!inpoly[0], "center of hole should be outside");
        assert!(inpoly[1], "between hole and edge should be inside");
        assert!(inpoly[2], "below hole should be inside");
        assert!(!inpoly[3], "outside diamond should be outside");
    }

    #[test]
    fn empty_polygon() {
        let node: Array2<f64> = Array2::zeros((0, 2));
        let point = array![[0.5, 0.5]];
        let (inpoly, onedge) = inpoly2(&point, &node, None, None);
        assert!(!inpoly[0], "empty polygon should contain nothing");
        assert!(!onedge[0]);
    }

    #[test]
    fn multiple_points() {
        let node = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let points = array![
            [0.5, 0.5],  // inside
            [2.0, 2.0],  // outside
            [0.5, 0.0],  // on edge
            [-0.5, 0.5], // outside
        ];
        let (inpoly, onedge) = inpoly2(&points, &node, None, None);

        assert!(inpoly[0]);
        assert!(!inpoly[1]);
        assert!(inpoly[2]);
        assert!(!inpoly[3]);

        assert!(!onedge[0]);
        assert!(!onedge[1]);
        assert!(onedge[2]);
        assert!(!onedge[3]);
    }

    #[test]
    fn triangle() {
        let node = array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let points = array![
            [0.5, 0.3],  // inside
            [0.5, 0.5],  // inside
            [0.1, 0.1],  // inside
            [0.9, 0.9],  // outside
        ];
        let (inpoly, _) = inpoly2(&points, &node, None, None);

        assert!(inpoly[0]);
        assert!(inpoly[1]);
        assert!(inpoly[2]);
        assert!(!inpoly[3]);
    }
}
