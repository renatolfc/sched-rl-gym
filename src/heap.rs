use std::collections::hash_map::DefaultHasher;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;

use pyo3::types::PyType;

fn py_hash(py: Python<'_>, obj: &Py<PyAny>) -> u64 {
    let mut hasher = DefaultHasher::new();
    match obj.bind(py).hash() {
        Ok(h) => h.hash(&mut hasher),
        Err(_) => obj.as_ptr().hash(&mut hasher),
    }
    hasher.finish()
}

fn py_eq(py: Python<'_>, a: &Py<PyAny>, b: &Py<PyAny>) -> bool {
    a.bind(py).eq(b.bind(py)).unwrap_or(false)
}

struct HeapEntry {
    priority: (i64, i64),
    counter: u64,
    generation: u64,
    key_hash: u64,
    item: Py<PyAny>,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.counter == other.counter
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (other.priority, other.counter).cmp(&(self.priority, self.counter))
    }
}

struct FinderEntry {
    key_hash: u64,
    item: Py<PyAny>,
    generation: u64,
}

#[pyclass(name = "Heap", module = "schedgym._schedgym_rs")]
pub struct PyHeap {
    pq: BinaryHeap<HeapEntry>,
    finder: Vec<FinderEntry>,
    counter: u64,
    generation: u64,
}

#[pymethods]
impl PyHeap {
    #[new]
    fn new() -> Self {
        PyHeap {
            pq: BinaryHeap::new(),
            finder: Vec::new(),
            counter: 0,
            generation: 0,
        }
    }

    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        Ok(cls.clone().into_any().unbind())
    }

    #[pyo3(signature = (item, priority = None))]
    fn add(&mut self, py: Python<'_>, item: Py<PyAny>, priority: Option<&Bound<'_, PyAny>>) {
        let prio: (i64, i64) = match priority {
            None => (0, 0),
            Some(obj) => {
                if let Ok(tuple) = obj.extract::<(i64, i64)>() {
                    tuple
                } else if let Ok(val) = obj.extract::<i64>() {
                    (val, 0)
                } else {
                    (0, 0)
                }
            }
        };
        let h = py_hash(py, &item);
        self.remove_from_finder(py, h, &item);

        self.generation += 1;
        let gen = self.generation;

        self.finder.push(FinderEntry {
            key_hash: h,
            item: item.clone_ref(py),
            generation: gen,
        });

        self.pq.push(HeapEntry {
            priority: prio,
            counter: self.counter,
            generation: gen,
            key_hash: h,
            item,
        });
        self.counter += 1;
    }

    fn remove(&mut self, py: Python<'_>, item: Py<PyAny>) -> PyResult<()> {
        let h = py_hash(py, &item);
        if !self.remove_from_finder(py, h, &item) {
            return Err(PyKeyError::new_err("Item not found in heap"));
        }
        Ok(())
    }

    fn pop(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        while let Some(entry) = self.pq.pop() {
            if self.is_live(py, &entry) {
                self.remove_from_finder(py, entry.key_hash, &entry.item);
                return Ok(entry.item);
            }
        }
        Err(PyKeyError::new_err("pop from an empty priority queue"))
    }

    #[getter]
    fn first(&mut self, py: Python<'_>) -> Option<PyObject> {
        loop {
            match self.pq.peek() {
                None => return None,
                Some(top) => {
                    if self.is_live(py, top) {
                        return Some(top.item.clone_ref(py));
                    }
                    self.pq.pop();
                }
            }
        }
    }

    fn heapsort(&self, py: Python<'_>) -> HeapSortIter {
        let mut entries: Vec<&HeapEntry> = self.pq.iter().filter(|e| self.is_live(py, e)).collect();
        entries.sort_by(|a, b| (a.priority, a.counter).cmp(&(b.priority, b.counter)));
        let items = entries.into_iter().map(|e| e.item.clone_ref(py)).collect();
        HeapSortIter { items, index: 0 }
    }

    fn __iter__(&self, py: Python<'_>) -> HeapSortIter {
        self.heapsort(py)
    }

    fn __contains__(&self, py: Python<'_>, item: Py<PyAny>) -> bool {
        let h = py_hash(py, &item);
        self.finder
            .iter()
            .any(|f| f.key_hash == h && py_eq(py, &f.item, &item))
    }

    fn __len__(&self) -> usize {
        self.finder.len()
    }

    fn __copy__(&self, py: Python<'_>) -> Self {
        let new_pq: BinaryHeap<HeapEntry> = self
            .pq
            .iter()
            .map(|e| HeapEntry {
                priority: e.priority,
                counter: e.counter,
                generation: e.generation,
                key_hash: e.key_hash,
                item: e.item.clone_ref(py),
            })
            .collect();
        let new_finder: Vec<FinderEntry> = self
            .finder
            .iter()
            .map(|f| FinderEntry {
                key_hash: f.key_hash,
                item: f.item.clone_ref(py),
                generation: f.generation,
            })
            .collect();
        PyHeap {
            pq: new_pq,
            finder: new_finder,
            counter: self.counter,
            generation: self.generation,
        }
    }

    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(&self, py: Python<'_>, _memo: Option<&Bound<'_, PyAny>>) -> Self {
        self.__copy__(py)
    }
}

impl PyHeap {
    fn is_live(&self, py: Python<'_>, entry: &HeapEntry) -> bool {
        self.finder.iter().any(|f| {
            f.generation == entry.generation
                && f.key_hash == entry.key_hash
                && py_eq(py, &f.item, &entry.item)
        })
    }

    fn remove_from_finder(&mut self, py: Python<'_>, h: u64, item: &Py<PyAny>) -> bool {
        if let Some(pos) = self
            .finder
            .iter()
            .position(|f| f.key_hash == h && py_eq(py, &f.item, item))
        {
            self.finder.swap_remove(pos);
            return true;
        }
        false
    }
}

#[pyclass(name = "HeapSortIter", module = "schedgym._schedgym_rs")]
pub struct HeapSortIter {
    items: Vec<Py<PyAny>>,
    index: usize,
}

#[pymethods]
impl HeapSortIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>, py: Python<'_>) -> Option<PyObject> {
        if slf.index < slf.items.len() {
            let item = slf.items[slf.index].clone_ref(py);
            slf.index += 1;
            Some(item)
        } else {
            None
        }
    }
}
