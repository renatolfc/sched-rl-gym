use crate::interval::PyInterval;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "IntervalTree", module = "schedgym.interval_tree")]
pub struct PyIntervalTree {
    pub intervals: Vec<PyInterval>,
}

#[pymethods]
impl PyIntervalTree {
    #[new]
    #[pyo3(signature = (intervals = None))]
    pub fn new(intervals: Option<Vec<PyRef<'_, PyInterval>>>, py: Python<'_>) -> Self {
        let ivs = match intervals {
            Some(vec) => vec.iter().map(|iv| iv.clone_ref(py)).collect(),
            None => Vec::new(),
        };
        let mut tree = PyIntervalTree { intervals: ivs };
        tree.intervals.sort_by_key(|iv| iv.begin);
        tree
    }

    pub fn add(&mut self, interval: &PyInterval, py: Python<'_>) {
        let iv = interval.clone_ref(py);
        let pos = self.intervals.partition_point(|x| x.begin < iv.begin);
        self.intervals.insert(pos, iv);
    }

    pub fn remove(&mut self, interval: &PyInterval, py: Python<'_>) -> PyResult<()> {
        for (i, iv) in self.intervals.iter().enumerate() {
            let eq = if iv.begin == interval.begin && iv.end == interval.end {
                match (&iv.data, &interval.data) {
                    (Some(d1), Some(d2)) => d1.bind(py).eq(d2.bind(py))?,
                    (None, None) => true,
                    _ => false,
                }
            } else {
                false
            };

            if eq {
                self.intervals.remove(i);
                return Ok(());
            }
        }
        Err(PyValueError::new_err("Interval not found"))
    }

    pub fn chop(&mut self, begin: i64, end: i64, py: Python<'_>) {
        let mut new_intervals = Vec::with_capacity(self.intervals.len());
        for iv in self.intervals.drain(..) {
            if iv.end <= begin || iv.begin >= end {
                // No overlap
                new_intervals.push(iv);
            } else if iv.begin >= begin && iv.end <= end {
                // Fully inside, remove entirely
            } else if iv.begin < begin && iv.end > end {
                // Spans across [begin, end), split into two
                new_intervals.push(PyInterval {
                    begin: iv.begin,
                    end: begin,
                    data: iv.data.as_ref().map(|d| d.clone_ref(py)),
                });
                new_intervals.push(PyInterval {
                    begin: end,
                    end: iv.end,
                    data: iv.data,
                });
            } else if iv.begin < begin {
                // Partially overlaps on the right (trim end)
                new_intervals.push(PyInterval {
                    begin: iv.begin,
                    end: begin,
                    data: iv.data,
                });
            } else {
                // Partially overlaps on the left (trim begin)
                new_intervals.push(PyInterval {
                    begin: end,
                    end: iv.end,
                    data: iv.data,
                });
            }
        }
        self.intervals = new_intervals;
    }

    pub fn merge_overlaps(&mut self, py: Python<'_>) {
        if self.intervals.is_empty() {
            return;
        }
        self.intervals.sort_by_key(|iv| iv.begin);
        let mut merged = Vec::with_capacity(self.intervals.len());
        let mut current = self.intervals[0].clone_ref(py);

        for iv in self.intervals.iter().skip(1) {
            if iv.begin <= current.end {
                // Overlaps or adjacent
                current.end = current.end.max(iv.end);
            } else {
                merged.push(current.clone_ref(py));
                current = iv.clone_ref(py);
            }
        }
        merged.push(current);
        self.intervals = merged;
    }

    pub fn begin(&self) -> PyResult<i64> {
        if self.intervals.is_empty() {
            Ok(i64::MAX)
        } else {
            Ok(self
                .intervals
                .iter()
                .map(|iv| iv.begin)
                .min()
                .unwrap_or(i64::MAX))
        }
    }

    pub fn end(&self) -> PyResult<i64> {
        if self.intervals.is_empty() {
            Ok(i64::MIN)
        } else {
            Ok(self
                .intervals
                .iter()
                .map(|iv| iv.end)
                .max()
                .unwrap_or(i64::MIN))
        }
    }

    fn __len__(&self) -> usize {
        self.intervals.len()
    }

    fn __bool__(&self) -> bool {
        !self.intervals.is_empty()
    }

    pub fn __contains__(&self, interval: &PyInterval, py: Python<'_>) -> PyResult<bool> {
        for iv in &self.intervals {
            let eq = if iv.begin == interval.begin && iv.end == interval.end {
                match (&iv.data, &interval.data) {
                    (Some(d1), Some(d2)) => d1.bind(py).eq(d2.bind(py))?,
                    (None, None) => true,
                    _ => false,
                }
            } else {
                false
            };
            if eq {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<IntervalTreeIter>> {
        let intervals = slf.intervals.iter().map(|iv| iv.clone_ref(py)).collect();
        let iter = IntervalTreeIter {
            intervals,
            index: 0,
        };
        Py::new(py, iter)
    }

    pub fn __repr__(&self, py: Python<'_>) -> String {
        let mut reprs = Vec::with_capacity(self.intervals.len());
        for iv in &self.intervals {
            let data_str = match &iv.data {
                Some(d) => d
                    .bind(py)
                    .repr()
                    .map(|r| r.to_string())
                    .unwrap_or_else(|_| "None".to_string()),
                None => "None".to_string(),
            };
            reprs.push(format!("Interval({}, {}, {})", iv.begin, iv.end, data_str));
        }
        format!("IntervalTree([{}])", reprs.join(", "))
    }

    fn __copy__(&self, py: Python<'_>) -> Self {
        PyIntervalTree {
            intervals: self.intervals.iter().map(|iv| iv.clone_ref(py)).collect(),
        }
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, py: Python<'_>, _memo: Option<&Bound<'_, PyAny>>) -> Self {
        self.__copy__(py)
    }
}

#[pyclass(name = "IntervalTreeIter", module = "schedgym.interval_tree")]
pub struct IntervalTreeIter {
    intervals: Vec<PyInterval>,
    index: usize,
}

#[pymethods]
impl IntervalTreeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Option<PyInterval> {
        if slf.index < slf.intervals.len() {
            let item = slf.intervals[slf.index].clone_ref(py);
            slf.index += 1;
            Some(item)
        } else {
            None
        }
    }
}
