use crate::interval::PyInterval;
use crate::interval_tree::PyIntervalTree;
use pyo3::exceptions::PyAssertionError;
use pyo3::prelude::*;

#[pyclass(name = "ResourcePool", module = "schedgym._schedgym_rs")]
pub struct PyResourcePool {
    #[pyo3(get)]
    pub size: i64,
    #[pyo3(get)]
    pub resource_type: i64,
    #[pyo3(get, set)]
    pub used_resources: i64,
    #[pyo3(get)]
    pub used_pool: Py<PyIntervalTree>,
}

#[pymethods]
impl PyResourcePool {
    #[new]
    #[pyo3(signature = (resource_type, size, used_pool = None))]
    fn new(
        py: Python<'_>,
        resource_type: i64,
        size: i64,
        used_pool: Option<Py<PyIntervalTree>>,
    ) -> PyResult<Self> {
        let pool_obj = match used_pool {
            Some(p) => {
                let used: i64 = {
                    let tree = p.borrow(py);
                    tree.intervals.iter().map(|iv| iv.end - iv.begin).sum()
                };
                return Ok(PyResourcePool {
                    size,
                    resource_type,
                    used_resources: used,
                    used_pool: p,
                });
            }
            None => Py::new(py, PyIntervalTree::new(None, py))?,
        };
        Ok(PyResourcePool {
            size,
            resource_type,
            used_resources: 0,
            used_pool: pool_obj,
        })
    }

    /// `.type` attribute alias — Python keyword requires `r#type` in Rust
    #[getter(r#type)]
    fn get_type(&self) -> i64 {
        self.resource_type
    }

    #[getter]
    fn free_resources(&self) -> i64 {
        self.size - self.used_resources
    }

    fn fits(&self, size: i64) -> PyResult<bool> {
        if size <= 0 {
            return Err(PyAssertionError::new_err("Can't allocate zero resources"));
        }
        Ok(size <= self.free_resources())
    }

    #[staticmethod]
    fn measure(interval: &PyInterval) -> i64 {
        interval.end - interval.begin
    }

    #[pyo3(signature = (size, data = None))]
    fn find(&self, py: Python<'_>, size: i64, data: Option<Py<PyAny>>) -> PyResult<PyIntervalTree> {
        if !self.fits(size)? {
            return Ok(PyIntervalTree::new(None, py));
        }

        let used_tree = self.used_pool.borrow(py);
        let mut free_intervals: Vec<PyInterval> = Vec::new();
        let mut cursor: i64 = 0;

        for iv in &used_tree.intervals {
            if iv.begin > cursor {
                free_intervals.push(PyInterval {
                    begin: cursor,
                    end: iv.begin,
                    data: data.as_ref().map(|d| d.clone_ref(py)),
                });
            }
            cursor = cursor.max(iv.end);
        }
        if cursor < self.size {
            free_intervals.push(PyInterval {
                begin: cursor,
                end: self.size,
                data: data.as_ref().map(|d| d.clone_ref(py)),
            });
        }

        let mut result = PyIntervalTree::new(None, py);
        let mut used_size: i64 = 0;

        for interval in &free_intervals {
            let measure = interval.end - interval.begin;
            let temp_size = measure + used_size;

            if temp_size == size {
                result.add(&interval.clone_ref(py), py);
                break;
            } else if temp_size < size {
                result.add(&interval.clone_ref(py), py);
                used_size = temp_size;
            } else {
                result.add(
                    &PyInterval {
                        begin: interval.begin,
                        end: interval.begin + size - used_size,
                        data: data.as_ref().map(|d| d.clone_ref(py)),
                    },
                    py,
                );
                break;
            }
        }

        Ok(result)
    }

    fn allocate(&mut self, py: Python<'_>, intervals: &Bound<'_, PyAny>) -> PyResult<()> {
        let iter = intervals.try_iter()?;
        for item in iter {
            let item = item?;
            let interval: PyRef<'_, PyInterval> = item.extract()?;
            let measure = interval.end - interval.begin;
            if self.used_resources + measure > self.size {
                return Err(PyAssertionError::new_err(
                    "Tried to allocate past size of resource pool",
                ));
            }
            {
                let mut tree = self.used_pool.borrow_mut(py);
                tree.add(&interval, py);
            }
            self.used_resources += measure;
        }
        Ok(())
    }

    fn free(&mut self, py: Python<'_>, intervals: &Bound<'_, PyAny>) -> PyResult<()> {
        let iter = intervals.try_iter()?;
        for item in iter {
            let item = item?;
            let interval: PyRef<'_, PyInterval> = item.extract()?;
            let measure = interval.end - interval.begin;
            {
                let tree = self.used_pool.borrow(py);
                if !tree.__contains__(&interval, py)? {
                    return Err(PyAssertionError::new_err(
                        "Tried to free unused resource set",
                    ));
                }
            }
            {
                let mut tree = self.used_pool.borrow_mut(py);
                tree.remove(&interval, py)?;
            }
            self.used_resources -= measure;
        }
        Ok(())
    }

    #[getter]
    fn intervals(&self, py: Python<'_>) -> Vec<PyInterval> {
        let tree = self.used_pool.borrow(py);
        tree.intervals.iter().map(|iv| iv.clone_ref(py)).collect()
    }

    fn clone(&self, py: Python<'_>) -> PyResult<Self> {
        let used_tree = self.used_pool.borrow(py);
        let cloned_intervals: Vec<PyInterval> = used_tree
            .intervals
            .iter()
            .map(|iv| iv.clone_ref(py))
            .collect();
        let new_tree = PyIntervalTree {
            intervals: cloned_intervals,
        };
        let new_pool = Py::new(py, new_tree)?;
        Ok(PyResourcePool {
            size: self.size,
            resource_type: self.resource_type,
            used_resources: self.used_resources,
            used_pool: new_pool,
        })
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let tree = self.used_pool.borrow(py);
        let tree_repr = tree.__repr__(py);
        format!(
            "ResourcePool(resource_type={}, size={}, used_pool={})",
            self.resource_type, self.size, tree_repr
        )
    }

    fn __copy__(&self, py: Python<'_>) -> PyResult<Self> {
        self.clone(py)
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, py: Python<'_>, _memo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        self.clone(py)
    }
}
