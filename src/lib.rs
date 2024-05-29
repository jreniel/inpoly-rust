use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray_stats::QuantileExt;

fn searchsorted_left(y_column: &[f64], y_values: &[f64]) -> Vec<usize> {
    y_values
        .iter()
        .map(|&y| {
            y_column
                .binary_search_by(|&val| val.partial_cmp(&y).unwrap())
                .unwrap_or_else(|x| x)
        })
        .collect()
}

fn searchsorted_right(y_column: &[f64], y_values: &[f64]) -> Vec<usize> {
    y_values
        .iter()
        .map(|&y| {
            match y_column.binary_search_by(|&val| val.partial_cmp(&y).unwrap()) {
                Ok(pos) => {
                    // Move to the next position for right side
                    let mut pos = pos;
                    while pos < y_column.len() && y_column[pos] == y {
                        pos += 1;
                    }
                    pos
                }
                Err(pos) => pos,
            }
        })
        .collect()
}

#[allow(non_snake_case)]
fn _inpoly(
    vert: &Array2<f64>,
    node: &Array2<f64>,
    edge: &Array2<usize>,
    ftol: &f64,
    lbar: &f64,
) -> (Array1<bool>, Array1<bool>) {
    // verbatim from the Python code, not entirely sure why the use of powf here.
    let feps = ftol * (lbar.powf(1.0));
    let veps = ftol * (lbar.powf(1.0));

    let vnum = vert.nrows();
    let mut stat = Array1::from_elem(vnum, false);
    let mut bnds = Array1::from_elem(vnum, false);

    // compute y-range overlap
    let XONE = node
        .select(Axis(0), &edge.column(0).to_owned().into_raw_vec())
        .column(0)
        .to_owned();
    let XTWO = node
        .select(Axis(0), &edge.column(1).to_owned().into_raw_vec())
        .column(0)
        .to_owned();
    let YONE = node
        .select(Axis(0), &edge.column(0).to_owned().into_raw_vec())
        .column(1)
        .to_owned();
    let YTWO = node
        .select(Axis(0), &edge.column(1).to_owned().into_raw_vec())
        .column(1)
        .to_owned();
    let XMIN: Vec<f64> = XONE
        .iter()
        .zip(XTWO.iter())
        .map(|(a, b)| a.min(*b))
        .collect();
    let XMAX: Vec<f64> = XONE
        .iter()
        .zip(XTWO.iter())
        .map(|(a, b)| a.max(*b))
        .collect();

    let XMIN: Vec<f64> = XMIN.iter().map(|x| x - veps).collect();
    let XMAX: Vec<f64> = XMAX.iter().map(|x| x + veps).collect();
    let YMIN = &YONE.mapv(|y| y - veps);
    let YMAX = &YTWO.mapv(|y| y + veps);

    let YDEL = &YTWO - &YONE;
    let XDEL = &XTWO - &XONE;

    let EDEL = &XDEL.mapv(|x| x.abs()) + &YDEL;

    // Get the sorted indices based on the second column (y values)
    let y_column = vert.column(1).to_vec();
    let mut ivec: Vec<usize> = (0..vert.nrows()).collect();
    ivec.sort_by(|&a, &b| y_column[a].partial_cmp(&y_column[b]).unwrap());
    let ione = searchsorted_left(&y_column, &YMIN.to_vec());
    let itwo = searchsorted_right(&y_column, &YMAX.to_vec());

    for epos in 0..edge.nrows() {
        let xone = XONE[epos];
        let xtwo = XTWO[epos];
        let yone = YONE[epos];
        let ytwo = YTWO[epos];
        let xmin = XMIN[epos];
        let xmax = XMAX[epos];
        let edel = EDEL[epos];
        let xdel = XDEL[epos];
        let ydel = YDEL[epos];

        for jpos in ione[epos]..itwo[epos] {
            let jvrt = ivec[jpos];
            if bnds[jvrt] == true {
                continue;
            }
            let xpos = vert[[jvrt, 0]];
            let ypos = vert[[jvrt, 1]];
            if xpos >= xmin {
                if xpos <= xmax {
                    // compute crossing number
                    let mul1 = ydel * (xpos - xone);
                    let mul2 = xdel * (ypos - yone);
                    if (feps * edel) >= (mul2 - mul1).abs() {
                        // BNDS -- approx. on edge
                        bnds[jvrt] = true;
                        stat[jvrt] = true;
                    } else if (ypos == yone) && (xpos == xone) {
                        // BNDS -- match about ONE
                        bnds[jvrt] = true;
                        stat[jvrt] = true;
                    } else if (ypos == ytwo) && (xpos == xtwo) {
                        // BNDS -- match about TWO
                        bnds[jvrt] = true;
                        stat[jvrt] = true;
                    } else if (mul1 <= mul2) && (ypos >= yone) && (ypos < ytwo) {
                        stat[jvrt] = !stat[jvrt]
                    }
                };
            } else if (ypos >= yone) && (ypos < ytwo) {
                // advance crossing number
                stat[jvrt] = !stat[jvrt]
            }
        }
    }
    (stat, bnds)
}

#[allow(non_snake_case)]
pub fn inpoly2(
    vert: &Array2<f64>,
    node: &Array2<f64>,
    edge: Option<&Array2<usize>>,
    ftol: Option<&f64>,
) -> (Array1<bool>, Array1<bool>) {
    let vnum = vert.nrows();

    let mut STAT = Array1::from_elem(vnum, false);
    let mut BNDS = Array1::from_elem(vnum, false);

    if node.is_empty() {
        return (STAT, BNDS);
    }
    let edge = match edge {
        None => {
            let n = node.nrows();
            let mut e = Array2::zeros((n, 2));
            for i in 0..n - 1 {
                e[[i, 0]] = i as usize;
                e[[i, 1]] = (i + 1) as usize;
            }
            e[[n - 1, 0]] = (n - 1) as usize;
            Some(e)
        }
        Some(edge) => Some(edge.clone()),
    };

    let edge = edge.unwrap();
    let used = edge.iter().map(|&e| e as usize).collect::<Vec<_>>();
    let sel0 = node.select(Axis(0), &used);
    let col0 = sel0.column(0);
    let xmin = col0.min().unwrap();
    let xmax = col0.max().unwrap();
    let col1 = sel0.column(1);
    let ymin = col1.min().unwrap();
    let ymax = col1.max().unwrap();
    let xdel = xmax - xmin;
    let ydel = ymax - ymin;
    let lbar = (xdel + ydel) / 2.0;
    let ftol = ftol.unwrap_or(&5.0e-14);
    let feps = ftol * lbar;
    // Mask to filter vertices
    let mask = vert
        .outer_iter()
        .map(|row| {
            row[0] >= xmin - feps
                && row[1] >= ymin - feps
                && row[0] <= xmax + feps
                && row[1] <= ymax + feps
        })
        .collect::<Vec<bool>>();
    let selected_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();
    let mut vert: Array2<f64> = vert.select(Axis(0), &selected_indices);
    if vert.is_empty() {
        return (STAT, BNDS);
    }

    // repeated code? xdel, ydel, lbar above were already computed from the subset
    // let xdel = vert.slice(s![.., 0]).max().unwrap() - vert.slice(s![.., 0]).min().unwrap();
    // let ydel = vert.slice(s![.., 1]).max().unwrap() - vert.slice(s![.., 1]).min().unwrap();
    // let lbar = (xdel + ydel) / 2.0;

    // flip to ensure y-axis is the `long` axis
    let mut node = node.clone();

    if xdel > ydel {
        vert.invert_axis(Axis(1));
        node.invert_axis(Axis(1));
    };

    // Sort edges by y-value
    let swap = edge
        .column(1)
        .iter()
        .zip(edge.column(0))
        .map(|(&y1, &y0)| node[(y1, 1)] < node[(y0, 1)])
        .collect::<Vec<bool>>();

    let mut edge = edge.clone();

    for (i, &swap) in swap.iter().enumerate() {
        if swap {
            edge.swap([i, 0], [i, 1]);
        }
    }
    let (stat, bnds) = _inpoly(&vert, &node, &edge, ftol, &lbar);
    for (i, &si) in selected_indices.iter().enumerate() {
        STAT[si] = stat[i];
        BNDS[si] = bnds[i];
    }
    (STAT, BNDS)
}

//
#[cfg(test)]
mod tests {

    use crate::inpoly2;
    use gnuplot::Caption;
    use gnuplot::Color;
    use gnuplot::Figure;
    use gnuplot::PointSymbol;
    use meshgridrs;
    use ndarray::array;
    use ndarray::stack;
    use ndarray::Array;
    use ndarray::Axis;

    #[test]
    fn example_1() {
        // node = np.array([
        //         [4, 0], [8, 4], [4, 8], [0, 4], [3, 3],
        //         [5, 3], [5, 5], [3, 5]])
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
        // edge = np.array([
        //     [0, 1], [1, 2], [2, 3], [3, 0], [4, 5],
        //     [5, 6], [6, 7], [7, 4]]);
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
        // xpos, ypos = np.meshgrid(
        //     np.linspace(-1, 9, 51), np.linspace(-1, 9, 51))
        let x = Array::linspace(-1., 9., 51);
        let y = Array::linspace(-1., 9., 51);
        let xi = vec![x, y];
        let grids = meshgridrs::meshgrid(&xi, meshgridrs::Indexing::Xy).unwrap();
        let xpos = &grids[0];
        let ypos = &grids[1];
        let xpos_flat = xpos.to_shape((xpos.len(), 1)).unwrap();
        let ypos_flat = ypos.to_shape((ypos.len(), 1)).unwrap();
        let points = stack![Axis(1), xpos_flat, ypos_flat].remove_axis(Axis(2));
        let (inpoly, onedge) = inpoly2(&points, &node, Some(&edge), None);
        let points_in: Vec<(f64, f64)> = points
            .outer_iter()
            .zip(&inpoly)
            .filter(|&(_, &in_set)| in_set)
            .map(|(row, _)| (row[0], row[1]))
            .collect();

        let points_out: Vec<(f64, f64)> = points
            .outer_iter()
            .zip(&inpoly)
            .filter(|&(_, &in_set)| !in_set)
            .map(|(row, _)| (row[0], row[1]))
            .collect();

        let points_on: Vec<(f64, f64)> = points
            .outer_iter()
            .zip(&onedge)
            .filter(|&(_, &on_set)| on_set)
            .map(|(row, _)| (row[0], row[1]))
            .collect();

        let x_in: Vec<f64> = points_in.iter().map(|&(x, _)| x).collect();
        let y_in: Vec<f64> = points_in.iter().map(|&(_, y)| y).collect();

        let x_out: Vec<f64> = points_out.iter().map(|&(x, _)| x).collect();
        let y_out: Vec<f64> = points_out.iter().map(|&(_, y)| y).collect();

        let x_on: Vec<f64> = points_on.iter().map(|&(x, _)| x).collect();
        let y_on: Vec<f64> = points_on.iter().map(|&(_, y)| y).collect();

        let mut fg = Figure::new();

        {
            let axes = fg.axes2d();
            axes
                //.set_aspect_ratio(1.0)
                .points(
                    &x_in,
                    &y_in,
                    &[Caption("IN==1"), Color("blue"), PointSymbol('O')],
                )
                .points(
                    &x_out,
                    &y_out,
                    &[Caption("IN==0"), Color("red"), PointSymbol('O')],
                )
                .points(
                    &x_on,
                    &y_on,
                    &[Caption("ON==1"), Color("magenta"), PointSymbol('S')],
                );
        }
        fg.show().unwrap();
    }
}
