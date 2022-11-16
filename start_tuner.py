import PyQt5.QtWidgets as wdg
import tuner
import sys

if __name__ == "__main__":
    app = wdg.QApplication(sys.argv)
    ex = tuner.TunerApp()
    sys.exit(app.exec_())
