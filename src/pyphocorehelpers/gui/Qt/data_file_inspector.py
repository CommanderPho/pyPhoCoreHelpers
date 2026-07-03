#!/usr/bin/env python
# coding: utf-8
"""Qt GUI for browsing a filesystem tree and inspecting .npy / .pkl file structure.

Usage:
    from pyphocorehelpers.gui.Qt.data_file_inspector import DataFileInspectorWindow, main
    main()

    # or:
    uv run python -m pyphocorehelpers.gui.Qt.data_file_inspector W:\\Data\\Bapun
"""
import os
import sys


def _fix_sys_path_for_stdlib_shadowing() -> None:
    """Drop cwd from sys.path when local pprint.py shadows the stdlib pprint module."""
    cwd = os.path.abspath(os.getcwd())
    if not os.path.isfile(os.path.join(cwd, 'pprint.py')):
        return
    sys.path[:] = [path_entry for path_entry in sys.path if path_entry != '' and os.path.abspath(path_entry) != cwd]


_fix_sys_path_for_stdlib_shadowing()
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import dill
import numpy as np
import pandas as pd
from qtpy import QtCore, QtGui, QtWidgets

from pyphocorehelpers.Filesystem.pickling_helpers import renamed_load

MAX_DEPTH = 8
MAX_CHILDREN = 200
INFO_TRUNCATE_LEN = 120
PREVIEW_MAX_CHARS = 4000
PREVIEW_ARRAY_MAX_ELEMS = 64
PREVIEW_DF_ROWS = 12
PREVIEW_SERIES_ROWS = 24
SUPPORTED_SUFFIXES = {'.npy', '.pkl'}
ROLE_OBJECT = QtCore.Qt.UserRole
ROLE_DEPTH = QtCore.Qt.UserRole + 1
ROLE_POPULATED = QtCore.Qt.UserRole + 2


def truncate_info(text: Any, max_len: int = INFO_TRUNCATE_LEN) -> str:
    text_str = str(text)
    if len(text_str) <= max_len:
        return text_str
    return text_str[:max_len - 3] + '...'


def format_type_summary(obj: Any) -> str:
    type_name = type(obj).__name__
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return type_name
    if isinstance(obj, np.ndarray):
        return f'ndarray shape={obj.shape}, dtype={obj.dtype}'
    if isinstance(obj, pd.DataFrame):
        return f'DataFrame shape={obj.shape}'
    if isinstance(obj, pd.Series):
        return f'Series len={len(obj)}, dtype={obj.dtype}'
    if isinstance(obj, dict):
        return f'dict len={len(obj)}'
    if isinstance(obj, list):
        return f'list len={len(obj)}'
    if isinstance(obj, tuple):
        return f'tuple len={len(obj)}'
    object_members = _get_object_members(obj)
    if object_members is not None:
        return f'{type_name} ({len(object_members)} attrs)'
    return type_name


def format_array_preview(arr: np.ndarray) -> str:
    if arr.size == 0:
        return '[]'
    if arr.ndim == 0:
        return repr(arr.item())
    if arr.ndim == 1:
        preview_values = arr.flat[:PREVIEW_ARRAY_MAX_ELEMS]
        return np.array2string(preview_values, threshold=PREVIEW_ARRAY_MAX_ELEMS, max_line_width=120)
    slice_tuple = tuple(slice(0, min(dim_size, 6)) for dim_size in arr.shape)
    preview_slice = arr[slice_tuple]
    suffix = f'\n... (showing corner slice of shape {preview_slice.shape}, full shape {arr.shape})' if preview_slice.shape != arr.shape else ''
    return np.array2string(preview_slice, threshold=PREVIEW_ARRAY_MAX_ELEMS, max_line_width=120) + suffix


def format_data_preview(obj: Any) -> str:
    if obj is None or isinstance(obj, (bool, int, float)):
        return repr(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        if len(obj) <= 64:
            return repr(obj)
        return repr(obj[:64]) + f' ... ({len(obj)} bytes total)'
    if isinstance(obj, np.ndarray):
        return format_array_preview(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.head(PREVIEW_DF_ROWS).to_string()
    if isinstance(obj, pd.Series):
        return obj.head(PREVIEW_SERIES_ROWS).to_string()
    if isinstance(obj, dict):
        lines = []
        for index, key in enumerate(sorted(obj.keys(), key=lambda item: str(item))[:PREVIEW_ARRAY_MAX_ELEMS]):
            lines.append(f'{repr(key)}: {truncate_info(repr(obj[key]), 80)}')
        if len(obj) > PREVIEW_ARRAY_MAX_ELEMS:
            lines.append(f'... ({len(obj) - PREVIEW_ARRAY_MAX_ELEMS} more keys)')
        return '\n'.join(lines)
    if isinstance(obj, (list, tuple)):
        lines = []
        for index, value in enumerate(obj[:PREVIEW_ARRAY_MAX_ELEMS]):
            lines.append(f'[{index}]: {truncate_info(repr(value), 80)}')
        if len(obj) > PREVIEW_ARRAY_MAX_ELEMS:
            lines.append(f'... ({len(obj) - PREVIEW_ARRAY_MAX_ELEMS} more items)')
        return '\n'.join(lines)
    object_members = _get_object_members(obj)
    if object_members is not None:
        lines = []
        for key in sorted(object_members.keys(), key=str)[:PREVIEW_ARRAY_MAX_ELEMS]:
            lines.append(f'{key}: {truncate_info(repr(object_members[key]), 80)}')
        if len(object_members) > PREVIEW_ARRAY_MAX_ELEMS:
            lines.append(f'... ({len(object_members) - PREVIEW_ARRAY_MAX_ELEMS} more attrs)')
        return '\n'.join(lines)
    return truncate_info(str(obj), PREVIEW_MAX_CHARS)


def _is_expandable(obj: Any, depth: int) -> bool:
    if depth >= MAX_DEPTH:
        return False
    if isinstance(obj, pd.DataFrame):
        return len(obj.columns) > 0
    if isinstance(obj, pd.Series):
        return len(obj) <= MAX_CHILDREN
    if isinstance(obj, dict):
        return len(obj) > 0
    if isinstance(obj, (list, tuple)):
        return len(obj) > 0
    if _get_object_members(obj) is not None:
        return True
    return False


def load_data_file(file_path: Path) -> Tuple[Any, str]:
    suffix = file_path.suffix.lower()
    if suffix == '.npy':
        return np.load(file_path, allow_pickle=True), 'np.load'
    if suffix == '.pkl':
        errors: List[str] = []
        try:
            with open(file_path, 'rb') as file_obj:
                return renamed_load(file_obj, move_modules_list={}), 'renamed_load'
        except Exception as exc:
            errors.append(f'renamed_load: {exc}')
        try:
            with open(file_path, 'rb') as file_obj:
                return dill.load(file_obj), 'dill.load'
        except Exception as exc:
            errors.append(f'dill.load: {exc}')
        try:
            with open(file_path, 'rb') as file_obj:
                return pickle.load(file_obj), 'pickle.load'
        except Exception as exc:
            errors.append(f'pickle.load: {exc}')
        raise RuntimeError('; '.join(errors))
    raise ValueError(f'Unsupported suffix: {suffix}')


def _get_object_members(obj: Any) -> Optional[dict]:
    members: dict = {}
    obj_dict = getattr(obj, '__dict__', None)
    if isinstance(obj_dict, dict):
        for key, value in obj_dict.items():
            if key.startswith('__') and key.endswith('__'):
                continue
            members[key] = value
    if not members:
        try:
            import attrs
            if attrs.has(type(obj)):
                for field in attrs.fields(type(obj)):
                    members[field.name] = getattr(obj, field.name, None)
        except ImportError:
            pass
    return members if members else None


def add_preview_item(tree: QtWidgets.QTreeWidget, parent_item: Optional[QtWidgets.QTreeWidgetItem], name: str, info: str) -> QtWidgets.QTreeWidgetItem:
    item = QtWidgets.QTreeWidgetItem([name, truncate_info(info)])
    if parent_item is None:
        tree.addTopLevelItem(item)
    else:
        parent_item.addChild(item)
    return item


def create_lazy_preview_item(tree: QtWidgets.QTreeWidget, parent_item: Optional[QtWidgets.QTreeWidgetItem], name: str, obj: Any, depth: int, extra_info: str = '') -> QtWidgets.QTreeWidgetItem:
    summary = format_type_summary(obj)
    short_preview = truncate_info(format_data_preview(obj), INFO_TRUNCATE_LEN)
    info_parts = [part for part in (summary, short_preview, extra_info) if part]
    item = add_preview_item(tree, parent_item, name, ' | '.join(info_parts))
    item.setData(0, ROLE_OBJECT, obj)
    item.setData(0, ROLE_DEPTH, depth)
    if _is_expandable(obj, depth):
        item.addChild(QtWidgets.QTreeWidgetItem(['', 'expand to load...']))
        item.setData(0, ROLE_POPULATED, False)
    else:
        item.setData(0, ROLE_POPULATED, True)
    return item


def populate_preview_children(parent_item: QtWidgets.QTreeWidgetItem, obj: Any, depth: int) -> None:
    tree = parent_item.treeWidget()
    if tree is None:
        return
    if isinstance(obj, pd.DataFrame):
        _add_lazy_container_children(tree, parent_item, list(obj.columns), lambda col: obj[col], depth)
        return
    if isinstance(obj, pd.Series):
        _add_lazy_container_children(tree, parent_item, list(obj.index[:MAX_CHILDREN]), lambda index: obj.loc[index], depth, key_fmt='[{k}]')
        return
    if isinstance(obj, dict):
        sorted_keys = sorted(obj.keys(), key=lambda key: str(key))
        _add_lazy_container_children(tree, parent_item, sorted_keys, lambda key: obj[key], depth)
        return
    if isinstance(obj, (list, tuple)):
        _add_lazy_container_children(tree, parent_item, list(range(len(obj))), lambda index: obj[index], depth, key_fmt='[{k}]')
        return
    object_members = _get_object_members(obj)
    if object_members is not None:
        sorted_member_keys = sorted(object_members.keys(), key=str)
        _add_lazy_container_children(tree, parent_item, sorted_member_keys, lambda key: object_members[key], depth)
        return


def _add_lazy_container_children(tree: QtWidgets.QTreeWidget, parent_item: QtWidgets.QTreeWidgetItem, keys: List[Any], getter: Callable[[Any], Any], depth: int, key_fmt: Optional[str] = None) -> None:
    total = len(keys)
    display_keys = keys[:MAX_CHILDREN]
    for key in display_keys:
        child_name = key_fmt.format(k=key) if key_fmt is not None else str(key)
        create_lazy_preview_item(tree, parent_item, child_name, getter(key), depth + 1)
    if total > MAX_CHILDREN:
        add_preview_item(tree, parent_item, f'... ({total - MAX_CHILDREN} more)', '')


def populate_lazy_preview_item_children(item: QtWidgets.QTreeWidgetItem) -> None:
    if item.data(0, ROLE_POPULATED):
        return
    obj = item.data(0, ROLE_OBJECT)
    depth = item.data(0, ROLE_DEPTH)
    if obj is None:
        item.setData(0, ROLE_POPULATED, True)
        return
    item.takeChildren()
    populate_preview_children(item, obj, depth)
    item.setData(0, ROLE_POPULATED, True)


class FileLoadWorker(QtCore.QObject):
    finished = QtCore.Signal(object, str, str)
    failed = QtCore.Signal(str, str)

    def __init__(self, file_path: Path):
        super().__init__()
        self._file_path = file_path
        self._cancelled = False


    @QtCore.Slot()
    def run(self) -> None:
        try:
            loaded_obj, loader_name = load_data_file(self._file_path)
            if not self._cancelled:
                self.finished.emit(loaded_obj, loader_name, str(self._file_path))
        except Exception as exc:
            if not self._cancelled:
                self.failed.emit(str(exc), str(self._file_path))


    def cancel(self) -> None:
        self._cancelled = True


class DataFileInspectorWindow(QtWidgets.QMainWindow):
    """Main window with filesystem tree (left) and loaded-object structure preview (right).

    Usage:

        uv run python -m pyphocorehelpers.gui.Qt.data_file_inspector H:/Data/Bapun/RatN/Day4OpenField/spykcirc/RatN_Day4_2019-10-15_11-30-06.GUI


    """

    def __init__(self, initial_root: Path):
        super().__init__()
        self._initial_root = initial_root.resolve()
        self._load_thread: Optional[QtCore.QThread] = None
        self._load_worker: Optional[FileLoadWorker] = None
        self._pending_file_path: Optional[Path] = None
        self.setWindowTitle('Data File Inspector')
        self.resize(1200, 800)
        self._build_ui()
        self._set_filesystem_root(self._initial_root)


    def _build_ui(self) -> None:
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        toolbar_layout = QtWidgets.QHBoxLayout()
        self._root_path_edit = QtWidgets.QLineEdit(str(self._initial_root))
        browse_button = QtWidgets.QPushButton('Browse')
        browse_button.clicked.connect(self._browse_root_folder)
        refresh_button = QtWidgets.QPushButton('Refresh')
        refresh_button.clicked.connect(self._refresh_filesystem_root)
        toolbar_layout.addWidget(QtWidgets.QLabel('Root:'))
        toolbar_layout.addWidget(self._root_path_edit, stretch=1)
        toolbar_layout.addWidget(browse_button)
        toolbar_layout.addWidget(refresh_button)
        main_layout.addLayout(toolbar_layout)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self._fs_model = QtWidgets.QFileSystemModel()
        self._fs_model.setFilter(QtCore.QDir.AllDirs | QtCore.QDir.AllEntries | QtCore.QDir.NoDotAndDotDot)
        self._fs_model.setRootPath('')
        self._fs_tree = QtWidgets.QTreeView()
        self._fs_tree.setModel(self._fs_model)
        self._fs_tree.setHeaderHidden(False)
        for column_index in (1, 2, 3):
            self._fs_tree.hideColumn(column_index)
        self._fs_tree.clicked.connect(self._on_filesystem_clicked)
        preview_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self._preview_tree = QtWidgets.QTreeWidget()
        self._preview_tree.setHeaderLabels(['Name', 'Summary'])
        self._preview_tree.setColumnWidth(0, 280)
        self._preview_tree.itemExpanded.connect(self._on_preview_item_expanded)
        self._preview_tree.itemSelectionChanged.connect(self._on_preview_selection_changed)
        self._preview_detail = QtWidgets.QTextEdit()
        self._preview_detail.setReadOnly(True)
        self._preview_detail.setPlaceholderText('Select a tree item to preview its data here.')
        self._preview_detail.setFont(QtGui.QFont('Consolas', 9))
        preview_splitter.addWidget(self._preview_tree)
        preview_splitter.addWidget(self._preview_detail)
        preview_splitter.setStretchFactor(0, 3)
        preview_splitter.setStretchFactor(1, 2)
        splitter.addWidget(self._fs_tree)
        splitter.addWidget(preview_splitter)
        splitter.setStretchFactor(0, 35)
        splitter.setStretchFactor(1, 65)
        main_layout.addWidget(splitter, stretch=1)
        self.statusBar().showMessage('Select a .npy or .pkl file to inspect.')


    def _set_filesystem_root(self, root_path: Path) -> None:
        resolved_root = root_path.resolve()
        if not resolved_root.exists():
            self.statusBar().showMessage(f'Root path does not exist: {resolved_root}')
            return
        root_path_str = str(resolved_root)
        self._fs_model.setRootPath(root_path_str)
        self._fs_tree.setRootIndex(self._fs_model.index(root_path_str))
        self._root_path_edit.setText(root_path_str)
        self.statusBar().showMessage(f'Browsing: {root_path_str}')


    def _browse_root_folder(self) -> None:
        selected_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Root Folder', self._root_path_edit.text())
        if selected_dir:
            self._set_filesystem_root(Path(selected_dir))


    def _refresh_filesystem_root(self) -> None:
        self._set_filesystem_root(Path(self._root_path_edit.text()))


    def _on_filesystem_clicked(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return
        file_path = Path(self._fs_model.filePath(index))
        if file_path.is_dir():
            self.statusBar().showMessage(f'Directory: {file_path}')
            return
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            self.statusBar().showMessage(f'Not a supported file: {file_path.name}')
            return
        self._start_load(file_path)


    def _cancel_active_load(self) -> None:
        if self._load_worker is not None:
            self._load_worker.cancel()
        if self._load_thread is not None and self._load_thread.isRunning():
            self._load_thread.quit()
            self._load_thread.wait(3000)


    def _cleanup_load_thread(self) -> None:
        if self._load_worker is not None:
            self._load_worker.deleteLater()
            self._load_worker = None
        if self._load_thread is not None:
            self._load_thread.deleteLater()
            self._load_thread = None


    def _start_load(self, file_path: Path) -> None:
        self._cancel_active_load()
        self._pending_file_path = file_path
        self.statusBar().showMessage(f'Loading {file_path.name}...')
        self._preview_tree.clear()
        add_preview_item(self._preview_tree, None, 'Loading...', str(file_path))
        self._load_thread = QtCore.QThread()
        self._load_worker = FileLoadWorker(file_path)
        self._load_worker.moveToThread(self._load_thread)
        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.finished.connect(self._on_load_finished)
        self._load_worker.failed.connect(self._on_load_failed)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.failed.connect(self._load_thread.quit)
        self._load_thread.finished.connect(self._cleanup_load_thread)
        self._load_thread.start()


    def _on_preview_item_expanded(self, item: QtWidgets.QTreeWidgetItem) -> None:
        populate_lazy_preview_item_children(item)


    def _on_preview_selection_changed(self) -> None:
        selected_items = self._preview_tree.selectedItems()
        if not selected_items:
            self._preview_detail.clear()
            return
        obj = selected_items[0].data(0, ROLE_OBJECT)
        if obj is None:
            self._preview_detail.setPlainText(selected_items[0].text(1))
            return
        preview_text = truncate_info(format_data_preview(obj), PREVIEW_MAX_CHARS)
        summary_text = format_type_summary(obj)
        self._preview_detail.setPlainText(f'{summary_text}\n{"=" * min(80, len(summary_text))}\n{preview_text}')


    def _on_load_finished(self, loaded_obj: Any, loader_name: str, file_path_str: str) -> None:
        if self._pending_file_path is None or str(self._pending_file_path) != file_path_str:
            return
        self._preview_tree.clear()
        self._preview_detail.clear()
        root_name = Path(file_path_str).name
        root_item = create_lazy_preview_item(self._preview_tree, None, root_name, loaded_obj, 0, extra_info=f'via {loader_name}')
        root_item.setExpanded(True)
        populate_lazy_preview_item_children(root_item)
        self._preview_tree.setCurrentItem(root_item)
        self._on_preview_selection_changed()
        self.statusBar().showMessage(f'Loaded {root_name} via {loader_name}')


    def _on_load_failed(self, error_message: str, file_path_str: str) -> None:
        if self._pending_file_path is None or str(self._pending_file_path) != file_path_str:
            return
        self._preview_tree.clear()
        add_preview_item(self._preview_tree, None, 'Load failed', truncate_info(error_message))
        self.statusBar().showMessage(f'Failed to load {Path(file_path_str).name}: {truncate_info(error_message)}')


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    app = QtWidgets.QApplication([])
    window = DataFileInspectorWindow(initial_root=root)
    window.show()
    result = app.exec()
    app.deleteLater()
    sys.exit(result)


if __name__ == '__main__':
    main()
