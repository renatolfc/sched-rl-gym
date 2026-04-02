use pyo3::class::basic::CompareOp;
use pyo3::prelude::*;
use std::hash::{Hash, Hasher};

#[pyclass(name = "Interval", module = "schedgym.interval")]
pub struct PyInterval {
    #[pyo3(get)]
    pub begin: i64,
    #[pyo3(get)]
    pub end: i64,
    #[pyo3(get)]
    pub data: Option<Py<PyAny>>,
}

impl PyInterval {
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            begin: self.begin,
            end: self.end,
            data: self.data.as_ref().map(|d| d.clone_ref(py)),
        }
    }
}

#[pymethods]
impl PyInterval {
    #[new]
    #[pyo3(signature = (begin, end, data = None))]
    pub fn new(begin: i64, end: i64, data: Option<Py<PyAny>>) -> Self {
        PyInterval { begin, end, data }
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.begin.hash(&mut hasher);
        self.end.hash(&mut hasher);
        hasher.finish()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp, py: Python<'_>) -> PyResult<PyObject> {
        match op {
            CompareOp::Eq => {
                let eq = if self.begin == other.begin && self.end == other.end {
                    match (&self.data, &other.data) {
                        (Some(d1), Some(d2)) => d1.bind(py).eq(d2.bind(py))?,
                        (None, None) => true,
                        _ => false,
                    }
                } else {
                    false
                };
                #[allow(deprecated)]
                Ok(eq.into_py(py))
            }
            CompareOp::Ne => {
                let eq = if self.begin == other.begin && self.end == other.end {
                    match (&self.data, &other.data) {
                        (Some(d1), Some(d2)) => d1.bind(py).eq(d2.bind(py))?,
                        (None, None) => true,
                        _ => false,
                    }
                } else {
                    false
                };
                #[allow(deprecated)]
                Ok((!eq).into_py(py))
            }
            _ => Ok(py.NotImplemented()),
        }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let data_str = match &self.data {
            Some(d) => d
                .bind(py)
                .repr()
                .map(|r| r.to_string())
                .unwrap_or_else(|_| "None".to_string()),
            None => "None".to_string(),
        };
        format!("Interval({}, {}, {})", self.begin, self.end, data_str)
    }

    fn __copy__(&self, py: Python<'_>) -> Self {
        self.clone_ref(py)
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, py: Python<'_>, _memo: Option<&Bound<'_, PyAny>>) -> Self {
        self.clone_ref(py)
    }
}
