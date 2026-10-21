#!/usr/bin/env python3
"""สร้างโปรเจกต์ QGIS พร้อม layout ของแผนที่สองแผ่น แล้ว export เป็นภาพ

  study_area   ที่ตั้งในประเทศไทย อำเภอของลำพูน และภาพ THEOS-2 สีจริงพร้อมขอบเขตการปกครอง
  landcover    แผนที่สิ่งปกคลุมดินของทั้งสองฉากจาก ensemble ที่เลือกจากชุดตรวจสอบ

ผู้อ่านวารสารด้านภูมิสารสนเทศคาดหวังกริดพิกัด ลูกศรเหนือ และมาตราส่วนแบบแผนที่
ซึ่งฉบับ matplotlib (fig_study_area.py, fig_map.py) ทำได้เพียงใกล้เคียง
โปรเจกต์ที่ได้เปิดแก้ต่อใน QGIS ได้ และรันสคริปต์นี้ซ้ำเพื่อสร้างใหม่จากข้อมูลได้เสมอ

ต้องรันด้วย Python ของ QGIS:
  QT_QPA_PLATFORM=offscreen /Applications/QGIS-final-4_2_2.app/Contents/MacOS/python \
      paper1/scripts/qgis_make_maps.py

ผลลัพธ์
  paper1/map/paper1_maps.qgz
  paper1/analysis/figure/manual/fig_study_area_qgis.png
  paper1/analysis/figure/manual/fig_map_qgis.png
"""
import argparse
import os
import sys

# PROJ ของ conda (miniforge) ชี้ proj.db คนละรุ่นกับของ QGIS ทำให้แปลงพิกัดล้มเหลวเงียบ ๆ
for _k in ("PROJ_DATA", "PROJ_LIB", "GDAL_DATA", "GDAL_DRIVER_PATH"):
    os.environ.pop(_k, None)
from pathlib import Path

from qgis.core import (QgsApplication, QgsCoordinateReferenceSystem, QgsFillSymbol,
                       QgsLayoutExporter, QgsLayoutItemLabel, QgsLayoutItemLegend,
                       QgsLayoutItemMap, QgsLayoutItemMapGrid, QgsLayoutItemPicture,
                       QgsLayoutItemScaleBar, QgsLayoutItemShape, QgsLayoutPoint,
                       QgsLayoutSize, QgsLegendStyle, QgsMarkerSymbol,
                       QgsPalLayerSettings, QgsPalettedRasterRenderer, QgsPrintLayout,
                       QgsProject, QgsRasterLayer, QgsRectangle, QgsRuleBasedRenderer,
                       QgsSingleSymbolRenderer, QgsTextBufferSettings, QgsTextFormat,
                       QgsUnitTypes, QgsVectorLayer, QgsVectorLayerSimpleLabeling,
                       QgsLayerTree, QgsContrastEnhancement, QgsMultiBandColorRenderer,
                       QgsMapLayerLegendUtils)
from qgis.PyQt.QtCore import QRectF
from qgis.PyQt.QtGui import QColor, QFont

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
SHAPES = ROOT / "analysis" / "shapes"
ADM = ROOT / "Landuse_lamphoon_2563" / "admin_boundary"
OUTDIR = ROOT / "analysis" / "figure" / "manual"
PROJECT = ROOT / "map" / "paper1_maps.qgz"
MM = QgsUnitTypes.LayoutMillimeters
UTM = QgsCoordinateReferenceSystem("EPSG:32647")
FONT = "Arial"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CLASS as CLASS_COLOR  # noqa: E402

CLASSES = ["background", "vegetation", "water", "urban", "agriculture", "bare_soil", "road"]
STRETCH = [(100, 541), (116, 395), (164, 404)]       # render.qml ชุดเดียวกับไทล์ฝึก
SCENE_COL = {"scene1": "#d55e00", "scene2": "#0072b2"}


def vec(path, name, encoding=None):
    lyr = QgsVectorLayer(str(path), name, "ogr")
    if not lyr.isValid():
        raise SystemExit(f"โหลดไม่ได้: {path}")
    if encoding:
        lyr.setProviderEncoding(encoding)
    return lyr


def fill(color, outline, width, style="solid"):
    return QgsFillSymbol.createSimple({"color": color, "outline_color": outline,
                                       "outline_width": str(width), "style": style})


def text_format(size, bold=False, buffer=True, color="#000000"):
    f = QgsTextFormat()
    font = QFont(FONT); font.setBold(bold)
    f.setFont(font); f.setSize(size); f.setColor(QColor(color))
    if buffer:
        b = QgsTextBufferSettings(); b.setEnabled(True); b.setSize(0.8)
        b.setColor(QColor("white")); f.setBuffer(b)
    return f


def label(layer, expr, size, bold=False, is_expr=True):
    s = QgsPalLayerSettings()
    s.fieldName = expr; s.isExpression = is_expr
    s.setFormat(text_format(size, bold))
    layer.setLabeling(QgsVectorLayerSimpleLabeling(s)); layer.setLabelsEnabled(True)


def layers():
    L = {}
    img = QgsRasterLayer(str(REPO / "theos" / "lamphoon" / "lamphoon_merged.tif"),
                         "THEOS-2 true colour (8 January 2026)")
    r = QgsMultiBandColorRenderer(img.dataProvider(), 1, 2, 3)
    for band, (lo, hi), setter in zip((1, 2, 3), STRETCH,
                                      (r.setRedContrastEnhancement, r.setGreenContrastEnhancement,
                                       r.setBlueContrastEnhancement)):
        ce = QgsContrastEnhancement(img.dataProvider().dataType(band))
        ce.setMinimumValue(lo); ce.setMaximumValue(hi)
        ce.setContrastEnhancementAlgorithm(QgsContrastEnhancement.ContrastEnhancementAlgorithm.StretchToMinimumMaximum)
        setter(ce)
    img.setRenderer(r)
    img.dataProvider().setNoDataValue(1, 0)
    L["img"] = img

    # lc1/lc2 = แผนที่เจ็ดคลาส (argmax ปกติ); lc61/lc62 = หกคลาส (ไม่ทำนาย background)
    for key, f, nm in (("lc1", "map_ensemble_lamphun.tif", "Land cover (seven classes), Scene 1"),
                       ("lc2", "map_ensemble_lamphun_scene2.tif", "Land cover (seven classes), Scene 2"),
                       ("lc61", "map_ensemble6_lamphun.tif", "Land cover (six classes), Scene 1"),
                       ("lc62", "map_ensemble6_lamphun_scene2.tif", "Land cover (six classes), Scene 2")):
        if not (ROOT / "outputs" / f).exists():
            continue
        ras = QgsRasterLayer(str(ROOT / "outputs" / f), nm)
        cls = [QgsPalettedRasterRenderer.Class(i, QColor(CLASS_COLOR[c]), c.replace("_", " "))
               for i, c in enumerate(CLASSES) if not (key in ("lc61", "lc62") and c == "background")]
        ras.setRenderer(QgsPalettedRasterRenderer(ras.dataProvider(), 1, cls))
        L[key] = ras

    tha = vec(SHAPES / "external" / "geoBoundaries-THA-ADM1_simplified.geojson", "Provinces of Thailand")
    root = QgsRuleBasedRenderer.Rule(None)
    for nm, flt, sym in (("Lamphun Province", "\"shapeName\" LIKE '%Lamphun%'",
                          fill("#c0392b", "#7b241c", 0.2)),
                         ("Other provinces", "ELSE", fill("#e6e6e6", "#9a9a9a", 0.1))):
        rule = QgsRuleBasedRenderer.Rule(sym, 0, 0, "" if flt == "ELSE" else flt, nm)
        if flt == "ELSE":
            rule.setIsElse(True)
        root.appendChild(rule)
    tha.setRenderer(QgsRuleBasedRenderer(root))
    L["tha"] = tha

    amp = vec(ADM / "amp_ลำพูน.shp", "District boundary")
    amp.setRenderer(QgsSingleSymbolRenderer(fill("#f4f4f4", "#555555", 0.25)))
    label(amp, "replace(replace(\"AMPHOE_E\", 'Amphoe ', ''), 'Muang', 'Mueang')", 5.5)
    L["amp_fill"] = amp
    amp2 = vec(ADM / "amp_ลำพูน.shp", "District boundary")
    amp2.setRenderer(QgsSingleSymbolRenderer(fill("0,0,0,0", "#111111", 0.5, "no")))
    L["amp_line"] = amp2
    tam = vec(ADM / "tambon_ลำพูน.shp", "Subdistrict boundary", "TIS-620")
    tam.setRenderer(QgsSingleSymbolRenderer(fill("0,0,0,0", "#ffd23f", 0.3, "no")))
    L["tam"] = tam
    wy = vec(SHAPES / "wiang_yong.geojson", "Wiang Yong Subdistrict")
    wy.setRenderer(QgsSingleSymbolRenderer(fill("0,0,0,0", "#ff2d55", 0.7, "no")))
    L["wy"] = wy
    lab = vec(SHAPES / "tambon_labels_en.geojson", "Subdistrict names")
    lab.setRenderer(QgsSingleSymbolRenderer(QgsMarkerSymbol.createSimple({"size": "0", "color": "0,0,0,0",
                                                                          "outline_style": "no"})))
    label(lab, "name_en", 6.5, is_expr=False)
    L["lab"] = lab
    # ฉบับ UTM 47N ของ scenes.geojson — ไม่ต้องแปลงพิกัดขณะวาด
    sc = vec(SHAPES / "scenes_utm47n.geojson", "THEOS-2 scene extents")
    root = QgsRuleBasedRenderer.Rule(None)
    for k, nm in (("scene1", "Scene 1 extent"), ("scene2", "Scene 2 extent")):
        root.appendChild(QgsRuleBasedRenderer.Rule(fill("0,0,0,0", SCENE_COL[k], 0.5, "no"),
                                                   0, 0, f"\"scene\" = '{k}'", nm))
    sc.setRenderer(QgsRuleBasedRenderer(root))
    L["scenes"] = sc
    return L


def add_map(layout, lyrs, x, y, w, h, extent, crs=UTM, frame=True):
    m = QgsLayoutItemMap(layout)
    m.attemptMove(QgsLayoutPoint(x, y, MM)); m.attemptResize(QgsLayoutSize(w, h, MM))
    m.setCrs(crs); m.setLayers(lyrs); m.setKeepLayerSet(True)
    m.zoomToExtent(extent)
    m.setFrameEnabled(frame); m.setFrameStrokeWidth(layout.convertFromLayoutUnits(0.3, MM))
    m.setBackgroundColor(QColor("white"))
    layout.addLayoutItem(m)
    return m


def add_grid(m, interval, unit_km=True, size=6):
    g = QgsLayoutItemMapGrid("grid", m)
    m.grids().addGrid(g)
    g.setIntervalX(interval); g.setIntervalY(interval)
    g.setStyle(QgsLayoutItemMapGrid.GridStyle.FrameAnnotationsOnly)
    g.setFrameStyle(QgsLayoutItemMapGrid.FrameStyle.ExteriorTicks)
    g.setFrameWidth(1.2); g.setFramePenSize(0.2)
    g.setAnnotationEnabled(True)
    g.setAnnotationFormat(QgsLayoutItemMapGrid.AnnotationFormat.CustomFormat)
    g.setAnnotationExpression("to_string(round(@grid_number / 1000))" if unit_km
                              else "to_string(@grid_number) || '°' || if(@grid_axis = 'x', 'E', 'N')")
    g.setAnnotationTextFormat(text_format(size, buffer=False))
    g.setAnnotationFrameDistance(1.0)
    V = QgsLayoutItemMapGrid.AnnotationDirection.Vertical
    for side in (QgsLayoutItemMapGrid.BorderSide.Left, QgsLayoutItemMapGrid.BorderSide.Right):
        g.setAnnotationDirection(V, side)
    g.setAnnotationDisplay(QgsLayoutItemMapGrid.DisplayMode.HideAll, QgsLayoutItemMapGrid.BorderSide.Right)
    g.setAnnotationDisplay(QgsLayoutItemMapGrid.DisplayMode.HideAll, QgsLayoutItemMapGrid.BorderSide.Top)
    return g


def add_label(layout, text, x, y, w, h, size=8, bold=False, align_center=False):
    lb = QgsLayoutItemLabel(layout)
    lb.setText(text); lb.setTextFormat(text_format(size, bold, buffer=False))
    lb.attemptMove(QgsLayoutPoint(x, y, MM)); lb.attemptResize(QgsLayoutSize(w, h, MM))
    if align_center:
        from qgis.PyQt.QtCore import Qt
        lb.setHAlign(Qt.AlignmentFlag.AlignHCenter)
    layout.addLayoutItem(lb)
    return lb


def add_scalebar(layout, m, x, y, km, segments=2):
    sb = QgsLayoutItemScaleBar(layout)
    sb.setStyle("Single Box"); sb.setLinkedMap(m)
    # หน่วยแผนที่เป็นเมตร: กำหนดความยาวช่วงเป็นเมตร แล้วให้ป้ายแสดงเป็นกิโลเมตร
    sb.setMapUnitsPerScaleBarUnit(1000); sb.setUnitLabel("km")
    sb.setNumberOfSegments(segments); sb.setNumberOfSegmentsLeft(0)
    sb.setUnitsPerSegment(km * 1000 / segments); sb.setHeight(1.5)
    sb.setTextFormat(text_format(6.5, buffer=False))
    sb.setBackgroundEnabled(True); sb.setBackgroundColor(QColor(255, 255, 255, 215))
    sb.attemptMove(QgsLayoutPoint(x, y, MM))
    layout.addLayoutItem(sb)
    return sb


def add_north(layout, x, y, size=11):
    svg = Path(QgsApplication.pkgDataPath()) / "svg" / "arrows" / "NorthArrow_02.svg"
    if not svg.exists():
        svg = Path("/Applications/QGIS-final-4_2_2.app/Contents/Resources/qgis/svg/arrows/NorthArrow_02.svg")
    print(f"north arrow: {svg} exists={svg.exists()}")
    p = QgsLayoutItemPicture(layout)
    p.setPicturePath(str(svg))
    p.attemptMove(QgsLayoutPoint(x, y, MM)); p.attemptResize(QgsLayoutSize(size * 0.7, size, MM))
    p.setBackgroundEnabled(True); p.setBackgroundColor(QColor(255, 255, 255, 215))
    layout.addLayoutItem(p)
    return p


def add_legend(layout, m, lyrs, x, y, columns=1, title=""):
    lg = QgsLayoutItemLegend(layout)
    lg.setLinkedMap(m); lg.setAutoUpdateModel(False)
    root = lg.model().rootGroup(); root.clear()
    for l in lyrs:
        root.addLayer(l)
    for node in root.findLayers():
        # ตัดบรรทัดชื่อแบนด์ ("Band 1 (Gray)") ของ raster แบบ paletted ออกจากคำอธิบายสัญลักษณ์
        if isinstance(node.layer(), QgsRasterLayer):
            n = len(lg.model().layerOriginalLegendNodes(node))
            QgsMapLayerLegendUtils.setLegendNodeOrder(node, list(range(1, n)))
            lg.model().refreshLayerLegend(node)
    lg.setTitle(title); lg.setColumnCount(columns); lg.setSplitLayer(True)
    for comp, size, bold in ((QgsLegendStyle.Style.SymbolLabel, 6.8, False),
                             (QgsLegendStyle.Style.Subgroup, 7, True),
                             (QgsLegendStyle.Style.Group, 7, True),
                             (QgsLegendStyle.Style.Title, 7.5, True)):
        st = lg.style(comp); st.setTextFormat(text_format(size, bold, buffer=False)); lg.setStyle(comp, st)
    lg.setSymbolWidth(5); lg.setSymbolHeight(3)
    lg.setFrameEnabled(False); lg.setBackgroundEnabled(False)
    lg.attemptMove(QgsLayoutPoint(x, y, MM))
    layout.addLayoutItem(lg); lg.adjustBoxSize()
    return lg


def new_layout(project, name, w, h):
    lay = QgsPrintLayout(project); lay.initializeDefaults(); lay.setName(name)
    lay.pageCollection().page(0).setPageSize(QgsLayoutSize(w, h, MM))
    project.layoutManager().addLayout(lay)
    return lay


def layout_study_area(project, L):
    lay = new_layout(project, "study_area", 182, 176)
    img_ext = L["img"].extent(); img_ext.grow(500)
    # (a) Thailand
    a = add_map(lay, [L["tha"]], 9, 9, 50, 70, QgsRectangle(97.2, 5.4, 105.9, 20.7),
                QgsCoordinateReferenceSystem("EPSG:4326"))
    g = add_grid(a, 4, unit_km=False, size=5.5)
    add_label(lay, "(a) Thailand", 9, 3.5, 50, 5, 8, True)
    # (b) Lamphun districts
    pe = L["amp_fill"].extent(); pe.grow(3000)
    b = add_map(lay, [L["scenes"], L["amp_fill"]], 9, 92, 50, 62, pe)
    add_grid(b, 40000, size=5.5)
    add_label(lay, "(b) Lamphun Province: districts", 9, 86.5, 55, 5, 8, True)
    add_scalebar(lay, b, 11, 143, 20)
    # (c) THEOS-2 true colour
    c = add_map(lay, [L["scenes"], L["lab"], L["wy"], L["amp_line"], L["tam"], L["img"]],
                68, 9, 108, 145, img_ext)
    add_grid(c, 4000, size=6)
    add_label(lay, "(c) THEOS-2 true colour, 8 January 2026 (UTM zone 47N, km)", 68, 3.5, 110, 5, 8, True)
    add_scalebar(lay, c, 71, 140, 4)
    add_north(lay, 71, 12)
    add_legend(lay, c, [L["tha"], L["amp_line"], L["tam"], L["wy"], L["scenes"]], 9, 158, columns=5)
    return lay


def layout_landcover(project, L, name, k1, k2, title, legend_name):
    """แผนที่สิ่งปกคลุมดินของทั้งสองฉาก; ในเขตทับซ้อนชั้นของฉากที่ 1 อยู่บน"""
    lay = new_layout(project, name, 150, 227)
    ext = L[k1].extent(); ext.combineExtentWith(L[k2].extent()); ext.grow(300)
    m = add_map(lay, [L["scenes"], L[k1], L[k2]], 12, 10, 130, 188, ext)
    add_grid(m, 4000, size=6.5)
    add_label(lay, title, 12, 3.5, 136, 5, 7.6, True)
    add_scalebar(lay, m, 15, 184, 4)
    add_north(lay, 130, 13)
    L[k1].setName(legend_name)
    add_legend(lay, m, [L[k1], L["scenes"]], 12, 203, columns=5)
    return lay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    a = ap.parse_args()
    QgsApplication.setPrefixPath("/Applications/QGIS-final-4_2_2.app/Contents/MacOS", True)
    app = QgsApplication([], False); app.initQgis()
    project = QgsProject.instance(); project.clear(); project.setCrs(UTM)
    project.setTitle("Paper 1 maps: study area and land cover")
    L = layers()
    for key in ("lc2", "lc1", "lc62", "lc61", "img", "tha", "amp_fill", "tam", "amp_line", "wy",
                "lab", "scenes"):
        if key in L:
            project.addMapLayer(L[key])
    lays = {"fig_study_area_qgis.png": layout_study_area(project, L)}
    when = "THEOS-2, 8 January 2026 (UTM zone 47N, km)"
    if "lc61" in L and "lc62" in L:
        # รูปหลัก: หกคลาส; แผนที่เจ็ดคลาสเก็บไว้เป็นรูปเสริม
        lays["fig_map_qgis.png"] = layout_landcover(
            project, L, "landcover", "lc61", "lc62",
            f"Land cover of both Lamphun scenes, six classes, {when}", "Land cover")
        lays["fig_map7_qgis.png"] = layout_landcover(
            project, L, "landcover7", "lc1", "lc2",
            f"Seven-class prediction including background, {when}", "Land cover")
    else:
        lays["fig_map_qgis.png"] = layout_landcover(
            project, L, "landcover", "lc1", "lc2",
            f"Land cover of both Lamphun scenes, {when}", "Land cover")
    PROJECT.parent.mkdir(parents=True, exist_ok=True)
    # เก็บ path แบบสัมพัทธ์เพื่อให้เปิดโปรเจกต์ได้เมื่อย้าย repository
    project.writeEntry("Paths", "/Absolute", False)
    project.write(str(PROJECT))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, lay in lays.items():
        s = QgsLayoutExporter.ImageExportSettings(); s.dpi = a.dpi
        res = QgsLayoutExporter(lay).exportToImage(str(OUTDIR / name), s)
        print(f"export {name}: {'ok' if res == QgsLayoutExporter.ExportResult.Success else res}")
    print(f"wrote {PROJECT}")
    app.exitQgis()


if __name__ == "__main__":
    main()
