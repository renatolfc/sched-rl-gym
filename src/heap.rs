use std::collections::{BinaryHeap, HashMap, HashSet};

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyType;

fn py_hash(py: Python<'_>, obj: &Py<PyAny>) -> isize {
    obj.bind(py).hash().unwrap_or(obj.as_ptr() as isize)
}

fn py_eq(py: Python<'_>, a: &Py<PyAny>, b: &Py<PyAny>) -> bool {
    a.bind(py).eq(b.bind(py)).unwrap_or(false)
}

struct HeapEntry {
    priority: (i64, i64),
    counter: u64,
    generation: u64,
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

#[pyclass(name = "Heap", module = "schedgym._schedgym_rs")]
pub struct PyHeap {
    pq: BinaryHeap<HeapEntry>,
    live_gens: HashSet<u64>,
    finder: HashMap<isize, Vec<(u64, Py<PyAny>)>>,
    len: usize,
    counter: u64,
    generation: u64,
}

#[pymethods]
impl PyHeap {
    #[new]
    fn new() -> Self {
        PyHeap {
            pq: BinaryHeap::new(),
            live_gens: HashSet::new(),
            finder: HashMap::new(),
            len: 0,
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
        self.remove_by_hash(py, h, &item);

        self.generation += 1;
        let gen = self.generation;

        self.live_gens.insert(gen);
        self.finder
            .entry(h)
            .or_default()
            .push((gen, item.clone_ref(py)));
        self.len += 1;

        self.pq.push(HeapEntry {
            priority: prio,
            counter: self.counter,
            generation: gen,
            item,
        });
        self.counter += 1;
    }

    fn remove(&mut self, py: Python<'_>, item: Py<PyAny>) -> PyResult<()> {
        let h = py_hash(py, &item);
        if !self.remove_by_hash(py, h, &item) {
            return Err(PyKeyError::new_err("Item not found in heap"));
        }
        Ok(())
    }

    fn pop(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        while let Some(entry) = self.pq.pop() {
            if self.live_gens.remove(&entry.generation) {
                self.remove_finder_by_gen(py, &entry.item, entry.generation);
                self.len -= 1;
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
                    if self.live_gens.contains(&top.generation) {
                        return Some(top.item.clone_ref(py));
                    }
                    self.pq.pop();
                }
            }
        }
    }

    fn heapsort(&self, py: Python<'_>) -> HeapSortIter {
        let mut entries: Vec<&HeapEntry> = self
            .pq
            .iter()
            .filter(|e| self.live_gens.contains(&e.generation))
            .collect();
        entries.sort_by(|a, b| (a.priority, a.counter).cmp(&(b.priority, b.counter)));
        let items = entries.into_iter().map(|e| e.item.clone_ref(py)).collect();
        HeapSortIter { items, index: 0 }
    }

    fn __iter__(&self, py: Python<'_>) -> HeapSortIter {
        self.heapsort(py)
    }

    fn __contains__(&self, py: Python<'_>, item: Py<PyAny>) -> bool {
        let h = py_hash(py, &item);
        match self.finder.get(&h) {
            None => false,
            Some(bucket) => bucket.iter().any(|(_, it)| py_eq(py, it, &item)),
        }
    }

    fn __len__(&self) -> usize {
        self.len
    }

    fn __copy__(&self, py: Python<'_>) -> Self {
        let new_pq: BinaryHeap<HeapEntry> = self
            .pq
            .iter()
            .map(|e| HeapEntry {
                priority: e.priority,
                counter: e.counter,
                generation: e.generation,
                item: e.item.clone_ref(py),
            })
            .collect();
        let new_finder: HashMap<isize, Vec<(u64, Py<PyAny>)>> = self
            .finder
            .iter()
            .map(|(&k, v)| {
                (
                    k,
                    v.iter().map(|(gen, it)| (*gen, it.clone_ref(py))).collect(),
                )
            })
            .collect();
        PyHeap {
            pq: new_pq,
            live_gens: self.live_gens.clone(),
            finder: new_finder,
            len: self.len,
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
    fn remove_by_hash(&mut self, py: Python<'_>, h: isize, item: &Py<PyAny>) -> bool {
        if let Some(bucket) = self.finder.get_mut(&h) {
            let pos = bucket.iter().position(|(_, it)| py_eq(py, it, item));
            if let Some(idx) = pos {
                let (gen, _) = bucket.swap_remove(idx);
                if bucket.is_empty() {
                    self.finder.remove(&h);
                }
                self.live_gens.remove(&gen);
                self.len -= 1;
                return true;
            }
        }
        false
    }

    fn remove_finder_by_gen(&mut self, _py: Python<'_>, item: &Py<PyAny>, generation: u64) {
        let h = py_hash(_py, item);
        if let Some(bucket) = self.finder.get_mut(&h) {
            if let Some(idx) = bucket.iter().position(|(g, _)| *g == generation) {
                bucket.swap_remove(idx);
                if bucket.is_empty() {
                    self.finder.remove(&h);
                }
            }
        }
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
