import sys
from utils import extract_ply_fields
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow)
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Qt, QFile 
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QStandardItemModel, QStandardItem, QWheelEvent


from PySide6.QtWidgets import  QGraphicsScene, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsPixmapItem
from PySide6.QtGui import QPen, QPixmap
from PySide6.QtCore import QObject, QPointF, QEvent

from PySide6.QtWidgets import QCheckBox, QGridLayout, QWidget 


class QStandardItemReadOnly(QStandardItem):
    def __init__(self, text):
        super().__init__(text)
        self.setEditable(False)  # 设置为只读
        self.setSelectable(True)  # 设置为可选
        self.setEnabled(True)  # 确保它是启用的


class FileExplorer():
    def __init__(self, main_window, ui):
        ui.openFile.clicked.connect(self.open_file_dialog)
        self.treeViewModel = QStandardItemModel()
        self.treeViewModel.setHorizontalHeaderLabels(["文件名"])
        ui.treeView.setModel(self.treeViewModel)
        ui.treeView.selectionModel().selectionChanged.connect(self.on_selection_changed)
        self.ui = ui
        self.main_window = main_window
        self.files = []
        self.last_dir = "/home/cvrsg/zyh_workspace/ytj/KITTI_360/sequences/"

    def add_ply_file_item(self, file_path: Path):
        item = QStandardItemReadOnly(file_path.name)
        propertys = extract_ply_fields(file_path)
        for value in propertys:
            clild = QStandardItemReadOnly(f"属性: {value}")
            item.appendRow(clild)

        self.treeViewModel.appendRow(item)
    
    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui,
            "请选择一个文件",
            self.last_dir,                     # 默认目录，空则为上次目录或当前工作目录
            "LAS文件 (*.las);;所有文件 (*.*)"
        )
        if file_path:
            file_path = Path(file_path)
            self.last_dir = str(file_path.parent)
            if file_path.exists():
                print(f"已打开文件: {file_path}")
                self.add_ply_file_item(file_path)
                self.files.append(file_path)
            else:
                print("File does not exist.")

    def on_selection_changed(self):
        indexes = self.ui.treeView.selectedIndexes()
        if indexes:
            index = indexes[0]
            if index.parent().isValid(): return
            item = self.treeViewModel.itemFromIndex(index)
            file_path = self.files[index.row()]
            self.main_window.pc_feature.init_pc_path(file_path)
            self.main_window.img_feature.init_pc_path(file_path)
            print(f"已选择文件: {item.text()}")
        else:
            print("No item selected.")


from PySide6.QtWidgets import QDialog


class ConfirmWidget(QDialog):
    def __init__(self, path):
        super().__init__()
        self.setWindowFlags(Qt.Dialog)
        self.resize(320, 120)

        qfile_ui = QFile("ui/confirm_widget.ui")
        qfile_ui.open(QFile.ReadOnly)
        self.ui = QUiLoader().load(qfile_ui)
        self.setLayout(self.ui.layout())
        qfile_ui.close()

        self.ui.path.setText(str(path))
        self.ui.confirm.clicked.connect(self.on_confirm)


    def on_confirm(self):
        path = self.ui.path.text()
        if not path:
            print("路径不能为空")
            return
        path = Path(path)
        self.accept()


# class ImageContainer()

# class DrawingPad():

# class HeightSelector(QWidget):


class StationFeature(QObject):
    def __init__(self, pc_file_path: Path, main_window, ui, ortho_img, side_img, info, side_info):
        super().__init__()
        self.main_window = main_window
        self.pc_file_path = pc_file_path
        self.info = info

        # for top view
        self.scene = QGraphicsScene()
        ortho_pixmap = QPixmap().fromImage(ortho_img)
        item = QGraphicsPixmapItem(ortho_pixmap)
        # item.setOffset(-ortho_pixmap.width() / 2, -ortho_pixmap.height() / 2)
        self.scene.addItem(item)
        # self.scene.setSceneRect(-ortho_pixmap.width()/2, -ortho_pixmap.height()/2,
        #                         ortho_pixmap.width(), ortho_pixmap.height())
        self.view = ui.topView
        self.view.installEventFilter(self)
        self.view.viewport().installEventFilter(self)

        self.view.setScene(self.scene)
        # self.view.centerOn(0, 0)
        self.view.setMouseTracking(True)
        self.view.viewport().setMouseTracking(True)

        self.points = []
        self.preview_line = None
        self._zoom_factor = 1.15
        self.is_dragging = False
        self._pan_start = QPointF()
        self._drag_pan_start = QPointF()
        self.line_color = Qt.red

        # for bot view
        self.bot_scene = QGraphicsScene()
        side_pixmap = QPixmap(side_img)
        item = QGraphicsPixmapItem(side_pixmap)
        # item.setOffset(-side_pixmap.width() / 2, -side_pixmap.height() / 2)
        self.bot_scene.addItem(item)
        # self.bot_scene.setSceneRect(-side_pixmap.width()/2, -side_pixmap.height()/2,
        #                             side_pixmap.width(), side_pixmap.height())
        self.bot_view = ui.botView
        self.bot_view.setScene(self.bot_scene)
        # self.bot_view.centerOn(0, 0)
        self.bot_view.setMouseTracking(True)
        self.bot_view.viewport().setMouseTracking(True)

        # top bar
        ui.startDrawing.clicked.connect(self.on_start_drawing)
        ui.stopDrawing.clicked.connect(self.on_stop_drawing)
        ui.exportStationFile.clicked.connect(self.on_return_pressed)

        self.ui = ui
        self.is_drawing = False


    def on_start_drawing(self):
        self.is_drawing = True
        print("开始绘图")


    def on_stop_drawing(self):
        self.is_drawing = False
        print("停止绘图")
        if self.preview_line:
            self.scene.removeItem(self.preview_line)
            self.preview_line = None


    def eventFilter(self, obj, event):        
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self.on_mouse_left_press(event)
            return True
        elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            self.on_mouse_right_press(event)
            return True
        elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
            self.on_mouse_right_release(event)
            return True
        elif event.type() == QEvent.MouseMove:
            self.on_mouse_move(event)
            return True
        elif event.type() == QEvent.KeyPress and event.key() == Qt.Key_Return:
            self.on_return_pressed()
            return True
        elif event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self.points.clear()
            self._pan_start = QPointF()
            self.preview_line = None
            for item in self.scene.items():
                if isinstance(item, QGraphicsLineItem) or isinstance(item, QGraphicsEllipseItem):
                    self.scene.removeItem(item)
            print("Cleared all points and lines.")
            return True
        elif event.type() == QEvent.Wheel:
            self.on_wheel(event)
            return True
        return False


    def on_wheel(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            zoom = self._zoom_factor
        else:
            zoom = 1 / self._zoom_factor
        self.view.scale(zoom, zoom)


    def on_mouse_left_press(self, event):
        if self.is_drawing:
            pos = self.view.mapToScene(event.position().toPoint())
            print(f"Mouse left pressed at: {pos.x()}, {pos.y()}")
            self.points.append((pos.x(), pos.y()))
            ellipse = QGraphicsEllipseItem(pos.x() - 3, pos.y() - 3, 6, 6)
            ellipse.setPen(QPen(Qt.red))
            ellipse.setBrush(Qt.black)
            self.scene.addItem(ellipse)
            self._pan_start = event.position()

            if len(self.points) > 1:
                p1 = QPointF(*self.points[-2])
                p2 = QPointF(*self.points[-1])
                line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
                line.setPen(QPen(self.line_color, 2))
                self.scene.addItem(line)


    def on_mouse_right_press(self, event):
        self.is_dragging = True
        self.view.setCursor(Qt.ClosedHandCursor)
        self._drag_pan_start = event.position()


    def on_mouse_right_release(self, event):
        self.is_dragging = False
        self.view.setCursor(Qt.ArrowCursor)


    def on_mouse_move(self, event):
        if self.points and self.is_drawing:
            p1 = QPointF(*self.points[-1])
            p2 = self.view.mapToScene(event.position().toPoint())
            if self.preview_line:
                self.scene.removeItem(self.preview_line)
            self.preview_line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
            self.preview_line.setPen(QPen(self.line_color, 2, Qt.DashLine))
            self.scene.addItem(self.preview_line)

        if self.is_dragging:
            delta = event.position() - self._drag_pan_start
            self.view.horizontalScrollBar().setValue(self.view.horizontalScrollBar().value() - delta.x())
            self.view.verticalScrollBar().setValue(self.view.verticalScrollBar().value() - delta.y())
            self._drag_pan_start = event.position()


    def on_return_pressed(self):
        if self.is_drawing:
            print("请先结束绘图")
        else:
            print(f"共选中{len(self.points)}个点，", self.points)
            output_path = self.pc_file_path.parent / f"{self.pc_file_path.stem}_stations.json"
            height = self.ui.heightValue.value()
            stride = self.ui.strideValue.value()
            confirm_widget = ConfirmWidget(output_path)
            
            print(f"站点文件路径: {output_path}")

            if confirm_widget.exec() == QDialog.Accepted:
                print("用户点了确定，去写文件……")
                from pyIMS.utils.stations import end_draw
                end_draw(self.points, self.info, height= height, stride= stride, output_path= output_path)
                self.main_window.img_feature.init_station_path(output_path)
            else:
                print("用户取消")
            

class WorkSpace():
    def __init__(self, main_window, ui):
        self.main_window = main_window
        self.tab_widget = ui.workSpace
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.tabs = []

    def close_tab(self, index):
        if index >= 0:
            self.tab_widget.removeTab(index)
            print(f"Closed tab at index {index}")
        else:
            print("No valid tab index to close.")
    
    def add_drawing_tab(self, pc_file_path, title="新建绘图"):
        
        qfile_ui = QFile("ui/drawing_pad.ui")
        qfile_ui.open(QFile.ReadOnly)
        new_tab = QUiLoader().load(qfile_ui)
        qfile_ui.close()

        tab_index = self.tab_widget.addTab(new_tab, title)
        self.tab_widget.setCurrentIndex(tab_index)

        from pyIMS.utils.stations import get_ortho_img
        from pyIMS.utils.io import read_las
        from PIL import Image
        from PIL.ImageQt import ImageQt
        import json

        
        ortho_img_path = pc_file_path.parent / f"{pc_file_path.stem}_ortho.png"
        ortho_info_path = pc_file_path.parent / f"{pc_file_path.stem}_ortho.json"
        side_img_path = pc_file_path.parent / f"{pc_file_path.stem}_side.png"
        side_info_path = pc_file_path.parent / f"{pc_file_path.stem}_side.json"

        if not (ortho_img_path.exists() and side_img_path.exists() and ortho_info_path.exists() and side_info_path.exists()):
            print(f"开始生成正射图和侧视图，点云文件：{pc_file_path}")
            points, colors, intensities = read_las(pc_file_path)
            # colors = np.stack([intensities, intensities, intensities], axis=-1)
            ortho_img, height_img, ortho_info = get_ortho_img(points, colors, width= 2048, view= 'top')

            ortho_img = Image.fromarray(ortho_img)
            ortho_img.save(ortho_img_path)

            with open(ortho_info_path, 'w') as f:
                json.dump(ortho_info, f)
            
            side_img, height_img, side_info = get_ortho_img(points, colors, width= 2048, view= 'front')

            side_img = Image.fromarray(side_img)
            side_img.save(side_img_path)

            with open(side_info_path, 'w') as f:
                json.dump(side_info, f)
        else:
            ortho_img = Image.open(ortho_img_path)
            with open(ortho_info_path, 'r') as f:
                ortho_info = json.load(f)

            side_img = Image.open(side_img_path)
            with open(side_info_path, 'r') as f:
                side_info = json.load(f)
        
        ortho_img = ImageQt(ortho_img)
        side_img = ImageQt(side_img)

        drawing = StationFeature(pc_file_path, self.main_window, new_tab, ortho_img, side_img, ortho_info, side_info)
        self.tabs.append(drawing)
        print(f"新建绘图标签页：{title}")
        return tab_index


class MultiObjectsCheckBox():
    def __init__(self, objects, layout: QGridLayout, column_count= 3):
        self.checkboxes = []

        for obj in objects:
            checkbox = QCheckBox(obj)
            row = len(self.checkboxes) // column_count
            col = len(self.checkboxes) % column_count
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox, row, col)

    def get_selected_objects(self):
        return [cb.text() for cb in self.checkboxes if cb.isChecked()]


class PcModelFeature():
    def __init__(self, ui):
        ui.pcModelSelect.addItems(['superpoint Transformer', 'Randla-Net'])
        ui.pcDatasetSelect.addItems(['Custom Dataset'])
        ui.startPcInfer.clicked.connect(self.on_start_pc_infer)
        self.ui = ui
    

    def init_pc_path(self, path):
        self.ui.pcPcFilePath.setText(str(path))
        self.ui.pcOutputLabelFilePath.setText(str(path.parent / f"{path.stem}_pc_label.npz"))

    def on_start_pc_infer(self):
        file_path = Path(self.ui.pcPcFilePath.text())
        if not file_path.exists():
            print("请选择一个有效的点云文件！")
            return

        output_path = Path(self.ui.pcOutputLabelFilePath.text())
        if not output_path.parent.exists():
            print("输出路径不存在，请选择一个有效的输出路径！")
            return
        
        model_name = self.ui.pcModelSelect.currentText()
        dataset_name = self.ui.pcDatasetSelect.currentText()

        print(f"点云文件路径：{file_path}\n"
              f"选定配置：\n  模型选用：{model_name}\n  数据集选用：{dataset_name}\n"
              f"标签写出路径：{output_path}\n"
              "点云语义分割模型推理开始...")
        
        # pc_inference(file_path, output_path, model_name, dataset_name)


class ImgModelFeature():
    def __init__(self, main_window, ui):
        self.main_window = main_window
        ui.addDrawingTab.clicked.connect(self.on_add_drawing_tab)
        ui.startProject.clicked.connect(self.on_start_project)
        ui.imgModelSelect.addItems(['Mask2Former-Dinov2', 'Mask2Former-ResNet50'])
        ui.imgDatasetSelect.addItems(['Custom']) 
        ui.startImgInfer.clicked.connect(self.on_start_img_infer)
        self.ui = ui
    
    def on_add_drawing_tab(self):
        pc_file_path = Path(self.ui.stationPcFilePath.text())
        if not pc_file_path.exists():
            print("请先选择有效点云文件")
            return
        self.main_window.workspace.add_drawing_tab(pc_file_path)

    def init_pc_path(self, path):
        self.ui.inputStationFilePath.setText(str(path.parent / f"{path.stem}_stations.json"))
        self.ui.stationPcFilePath.setText(str(path))
        self.ui.imgPcFilePath.setText(str(path))
        self.ui.imgProjectedImgOutputDir.setText(str(path.parent / "projected_images"))
        self.ui.imgProjectedImgInputDir.setText(str(path.parent / "projected_images"))
        self.ui.imgOutputLabelFilePath.setText(str(path.parent / f"{path.stem}_img_label.npz"))
    
    def init_station_path(self, path):
        self.ui.inputStationFilePath.setText(str(path))


    def on_start_project(self):
        pc_file_path = Path(self.ui.imgPcFilePath.text())
        if not pc_file_path.exists():
            print("请选择一个有效的点云文件！")
            return
            
        import json
        stations_path = Path(self.ui.inputStationFilePath.text())
        if not stations_path.exists():
            print("请选择一个有效的站点文件！")
            return
        else:
            stations = json.load(open(stations_path, 'r'))['stations']

        output_dir = Path(self.ui.imgProjectedImgOutputDir.text())
        output_dir.mkdir(parents=True, exist_ok=True) 
        
        from pyIMS import pc_proj_preprocess
        from pyIMS.utils.io import read_las
        
        print(f"开始投影图片，点云文件：{pc_file_path}, 输出目录：{output_dir}")
        points, colors, intensities = read_las(pc_file_path)
        pc_proj_preprocess(points, colors, intensities, stations, output_dir, img_shape=(720, 1280))
        
        with open(output_dir / 'num_points.txt', 'w') as f:
            f.write(str(len(points)))



    def on_start_img_infer(self):
        img_dir_path = Path(self.ui.imgProjectedImgInputDir.text())
        if not img_dir_path.exists():
            print("请选择一个有效的图片目录！")
            return
        
        with open(img_dir_path / 'num_points.txt', 'r') as f:
            num_points = int(f.read().strip())


        output_path = Path(self.ui.imgOutputLabelFilePath.text())
        if not output_path.parent.exists():
            print("输出路径不存在，请选择一个有效的输出路径！")
            return
        
        model_name = self.ui.imgModelSelect.currentText()
        dataset_name = self.ui.imgDatasetSelect.currentText()
        is_output_mask = self.ui.isOutPutMaskCheckBox.isChecked()

        print(f"投影图片目录：{img_dir_path}\n"
              f"选定配置：\n  模型选用：{model_name}\n  数据集选用：{dataset_name}\n"
              f"标签写出路径：{output_path}\n"
              f"是否输出图片分割结果：{is_output_mask}\n"
              "图像语义分割模型推理开始...")
        
        from pyIMS import img_inference
        with open(img_dir_path / 'num_points.txt', 'r') as f:
            num_points = int(f.read().strip())
        img_inference(img_dir_path, num_points, output_path, is_output_mask)


class VecFeature():
    def __init__(self, ui):
        ui.startVec.clicked.connect(self.on_start_vec)

        vec_objects = ['路面', '道路标记', '杆状物', '建筑物']
        self.vec_checkbox_layout = MultiObjectsCheckBox(vec_objects, ui.vecObjectGridLayout)    
    
    def on_start_vec(self):
        objs = self.vec_checkbox_layout.get_selected_objects()
        print(f'开始矢量化，选中的对象: {objs}')


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能测绘系统")
        self.setGeometry(400, 300, 1400, 800)
        
        self.setup_ui()
     
        self.file_explorer = FileExplorer(self, self.ui)
        self.workspace = WorkSpace(self, self.ui)
        self.pc_feature = PcModelFeature(self.ui)
        self.img_feature = ImgModelFeature(self, self.ui)
        self.vec_feature = VecFeature(self.ui)

    def setup_ui(self):
        qfile_ui = QFile("ui/main.ui")
        qfile_ui.open(QFile.ReadOnly)
        self.ui = QUiLoader().load(qfile_ui)
        qfile_ui.close()
        self.setCentralWidget(self.ui)
            

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
