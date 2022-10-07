import sys
from PyQt5 import QtCore, QtXml

""" from https://stackoverflow.com/questions/52029649/qtdesigner-pyqt-pyside-disable-translatable-by-default 
Supposedly takes a .ui file produced by uic and QtDesigner and turns off the "Translatable" property, cleaning up the code.
UNTESTED

Other Discussion:
https://stackoverflow.com/questions/51483773/how-can-i-remove-the-translatable-checkbox-in-qt-designer-for-a-qstring-proper
https://doc.qt.io/qt-5/designer-creating-custom-widgets.html#notes-on-the-domxml-function
https://doc.qt.io/qt-5/qtdesigner-customwidgetplugin-example.html
https://doc.qt.io/qt-5/quiloader.html - The QUiLoader class enables standalone applications to dynamically create user interfaces at run-time using the information stored in UI files or specified in plugin paths.

"""

def modify_ui_file(filename="/path/of/your_file.ui"):

    file = QtCore.QFile(filename)
    if not file.open(QtCore.QFile.ReadOnly):
        sys.exit(-1)

    doc = QtXml.QDomDocument()
    if not doc.setContent(file):
        sys.exit(-1)
    file.close()

    strings = doc.elementsByTagName("string")
    for i in range(strings.count()):
        strings.item(i).toElement().setAttribute("notr", "true")

    if not file.open(QtCore.QFile.Truncate|QtCore.QFile.WriteOnly):
        sys.exit(-1)

    xml = doc.toByteArray()
    file.write(xml)
    file.close()

if __name__ == '__main__':

    filename = r"C:\Users\pho\repos\pyPhoPlaceCellAnalysis\src\pyphoplacecellanalysis\GUI\Qt\Menus\GlobalApplicationMenusMainWindow\GlobalApplicationMenusMainWindow.ui"
    modify_ui_file(filename=filename)