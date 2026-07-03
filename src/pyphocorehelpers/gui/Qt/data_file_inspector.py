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
from qtpy import QtCore, QtWidgets

from pyphocorehelpers.Filesystem.pickling_helpers import renamed_load

MAX_DEPTH = 8
MAX_CHILDREN = 200
INFO_TRUNCATE_LEN = 120
SUPPORTED_SUFFIXES = {'.npy', '.pkl'}


def truncate_info(text: Any) -> str:
    text_str = str(text)
    if len(text_str) <= INFO_TRUNCATE_LEN:
        return text_str
    return text_str[:INFO_TRUNCATE_LEN - 3] + '...'


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
    item = QtWidgets.QTreeWidgetItem([name, info])
    if parent_item is None:
        tree.addTopLevelItem(item)
    else:
        parent_item.addChild(item)
    return item


def _add_container_children(tree: QtWidgets.QTreeWidget, parent_item: QtWidgets.QTreeWidgetItem, keys: List[Any], getter: Callable[[Any], Any], depth: int, key_fmt: Optional[str] = None) -> None:
    total = len(keys)
    display_keys = keys[:MAX_CHILDREN]
    for key in display_keys:
        child_name = key_fmt.format(k=key) if key_fmt is not None else str(key)
        populate_preview_tree(tree, parent_item, getter(key), child_name, depth + 1)
    if total > MAX_CHILDREN:
        add_preview_item(tree, parent_item, f'... ({total - MAX_CHILDREN} more)', '')


def populate_preview_tree(tree: QtWidgets.QTreeWidget, parent_item: Optional[QtWidgets.QTreeWidgetItem], obj: Any, name: str, depth: int) -> QtWidgets.QTreeWidgetItem:
    type_name = type(obj).__name__
    if depth >= MAX_DEPTH:
        return add_preview_item(tree, parent_item, name, f'{type_name} (max depth reached)')
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return add_preview_item(tree, parent_item, name, truncate_info(repr(obj)))
    if isinstance(obj, np.ndarray):
        return add_preview_item(tree, parent_item, name, f'shape={obj.shape}, dtype={obj.dtype}')
    if isinstance(obj, pd.DataFrame):
        item = add_preview_item(tree, parent_item, name, f'DataFrame shape={obj.shape}')
        _add_container_children(tree, item, list(obj.columns), lambda col: obj[col], depth)
        return item
    if isinstance(obj, pd.Series):
        index_preview = ', '.join(str(index_label) for index_label in obj.index[:5])
        if len(obj) > 5:
            index_preview += ', ...'
        return add_preview_item(tree, parent_item, name, f'Series len={len(obj)}, dtype={obj.dtype}, index=[{index_preview}]')
    if isinstance(obj, dict):
        item = add_preview_item(tree, parent_item, name, f'dict len={len(obj)}')
        sorted_keys = sorted(obj.keys(), key=lambda key: str(key))
        _add_container_children(tree, item, sorted_keys, lambda key: obj[key], depth)
        return item
    if isinstance(obj, (list, tuple)):
        type_label = 'list' if isinstance(obj, list) else 'tuple'
        item = add_preview_item(tree, parent_item, name, f'{type_label} len={len(obj)}')
        _add_container_children(tree, item, list(range(len(obj))), lambda index: obj[index], depth, key_fmt='[{k}]')
        return item
    object_members = _get_object_members(obj)
    if object_members is not None:
        item = add_preview_item(tree, parent_item, name, f'{type_name} ({len(object_members)} attrs)')
        sorted_member_keys = sorted(object_members.keys(), key=str)
        _add_container_children(tree, item, sorted_member_keys, lambda key: object_members[key], depth)
        return item
    return add_preview_item(tree, parent_item, name, truncate_info(f'{type_name}: {obj}'))


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
        self._preview_tree = QtWidgets.QTreeWidget()
        self._preview_tree.setHeaderLabels(['Name', 'Info'])
        self._preview_tree.setColumnWidth(0, 320)
        splitter.addWidget(self._fs_tree)
        splitter.addWidget(self._preview_tree)
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


    def _on_load_finished(self, loaded_obj: Any, loader_name: str, file_path_str: str) -> None:
        if self._pending_file_path is None or str(self._pending_file_path) != file_path_str:
            return
        self._preview_tree.clear()
        root_name = Path(file_path_str).name
        root_type_name = type(loaded_obj).__name__
        root_item = add_preview_item(self._preview_tree, None, root_name, f'{root_type_name} via {loader_name}')
        populate_preview_tree(self._preview_tree, root_item, loaded_obj, 'value', depth=0)
        root_item.setExpanded(True)
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
