mod heap;
mod interval;
mod interval_tree;
mod pool;

use pyo3::prelude::*;

#[pymodule]
fn _schedgym_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<interval::PyInterval>()?;
    m.add_class::<interval_tree::PyIntervalTree>()?;
    m.add_class::<interval_tree::IntervalTreeIter>()?;
    m.add_class::<heap::PyHeap>()?;
    m.add_class::<heap::HeapSortIter>()?;
    m.add_class::<pool::PyResourcePool>()?;
    Ok(())
}
