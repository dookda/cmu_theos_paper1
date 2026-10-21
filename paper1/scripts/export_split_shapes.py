#!/usr/bin/env python3
"""ส่งออกรูปทรงของการแบ่งข้อมูลเชิงพื้นที่ เพื่อนำไปทำแผนที่ใน GIS

รูปที่ 2 ของบทความวาดด้วย matplotlib ซึ่งแก้ยากถ้าอยากได้แผนที่จริงจัง
สคริปต์นี้ส่งออกเรขาคณิตชุดเดียวกับที่รูปนั้นใช้ ให้เปิดใน QGIS/ArcGIS ได้

ชั้นข้อมูลที่ได้ (GeoPackage ไฟล์เดียว + GeoJSON แยกชั้น)
  blocks   บล็อกเชิงพื้นที่ 1500 ม. ทั้งภายในและภายนอก พร้อม split และจำนวนไทล์
  tiles    ไทล์ทุกใบ พร้อม split และธงว่าถูกตัดเป็นแนวกันชนหรือไม่
  scenes   กรอบของทั้งสองฉาก THEOS-2

CRS: EPSG:32647 (WGS 84 / UTM zone 47N) ตามที่ bounds ในไฟล์ต้นทางใช้

ใช้งาน
------
  python paper1/scripts/export_split_shapes.py
  python paper1/scripts/export_split_shapes.py --out-dir /tmp/shapes
"""
import argparse
import json
import os
from pathlib import Path

# PROJ_DATA ในเครื่องนี้ชี้ไปฐานข้อมูลรุ่นเก่าของ miniforge ทำให้ GDAL อ่าน EPSG ไม่ออก
# และเขียนไฟล์ออกมาโดยไม่มี CRS เลย ต้องบังคับทับก่อน import osgeo ไม่ใช่ setdefault
for _cand in ("/opt/homebrew/share/proj",
              "/usr/share/proj", "/usr/local/share/proj"):
    if os.path.exists(os.path.join(_cand, "proj.db")):
        os.environ["PROJ_DATA"] = _cand
        os.environ["PROJ_LIB"] = _cand
        break

from osgeo import ogr, osr                                   # noqa: E402

osr.UseExceptions()
ogr.UseExceptions()

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "analysis" / "results" / "spatial_splits.json"
META = ROOT.parent / "paper2" / "analysis" / "results" / "tiles_metadata.json"
EPSG = 32647


def ring(left, bottom, right, top):
    r = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ((left, bottom), (right, bottom), (right, top), (left, top),
                 (left, bottom)):
        r.AddPoint_2D(x, y)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(r)
    return poly


def load():
    sp = json.loads(SPLITS.read_text())
    meta = json.loads(META.read_text())
    bounds = {Path(m["filename"]).stem: m["bounds"] for m in meta}
    split_of = {t: s for s, ts in sp["splits"].items() for t in ts}
    dropped = {t: s for s, ts in (sp.get("buffer_dropped") or {}).items() for t in ts}
    return sp, bounds, split_of, dropped


def block_extent(tiles, bounds):
    bs = [bounds[t] for t in tiles if t in bounds]
    if not bs:
        return None
    return (min(b["left"] for b in bs), min(b["bottom"] for b in bs),
            max(b["right"] for b in bs), max(b["top"] for b in bs))


def build(out_dir):
    sp, bounds, split_of, dropped = load()
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)

    layers = {}

    # ---- blocks ----
    feats = []
    for src, ext in (("internal", sp["blocks"]), ("external", sp["blocks_external"])):
        for bid, info in ext.items():
            e = block_extent(info["tiles"], bounds)
            if e is None:
                continue
            feats.append((ring(*e), {
                "block_id": bid, "split": info["split"], "pool": src,
                "n_tiles": len(info["tiles"]),
            }))
    layers["blocks"] = (feats, {"block_id": ogr.OFTString, "split": ogr.OFTString,
                                "pool": ogr.OFTString, "n_tiles": ogr.OFTInteger})

    # ---- tiles ----
    feats = []
    for name, b in bounds.items():
        s = split_of.get(name)
        d = dropped.get(name)
        if s is None and d is None:
            continue                     # ไทล์ที่ไม่ได้ใช้เลย ไม่ต้องส่งออก
        feats.append((ring(b["left"], b["bottom"], b["right"], b["top"]), {
            "tile": name,
            "split": s or "",
            "buffer_dropped": 1 if d else 0,
            "dropped_from": d or "",
        }))
    layers["tiles"] = (feats, {"tile": ogr.OFTString, "split": ogr.OFTString,
                               "buffer_dropped": ogr.OFTInteger,
                               "dropped_from": ogr.OFTString})

    # ---- scenes ----
    feats = [(ring(*v), {"scene": k})
             for k, v in sp["scene_bounds_utm47n"].items()]
    layers["scenes"] = (feats, {"scene": ogr.OFTString})

    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / "spatial_splits.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    drv = ogr.GetDriverByName("GPKG")
    ds = drv.CreateDataSource(str(gpkg))
    for name, (feats, fields) in layers.items():
        lyr = ds.CreateLayer(name, srs, ogr.wkbPolygon)
        for fname, ftype in fields.items():
            lyr.CreateField(ogr.FieldDefn(fname, ftype))
        for geom, attrs in feats:
            f = ogr.Feature(lyr.GetLayerDefn())
            for k, v in attrs.items():
                f.SetField(k, v)
            f.SetGeometry(geom)
            lyr.CreateFeature(f)
            f = None
        print(f"  {name:<8} {len(feats):>5} features")
    ds = None
    print(f"wrote {gpkg}")

    # GeoJSON แยกชั้น สำหรับเครื่องมือที่ไม่อ่าน GeoPackage
    # สเปก GeoJSON กำหนดให้ใช้ WGS84 lon/lat จึงต้องแปลงพิกัด ไม่ใช่เขียน UTM ทิ้งไว้
    wgs = osr.SpatialReference()
    wgs.ImportFromEPSG(4326)
    wgs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tx = osr.CoordinateTransformation(srs, wgs)
    gdrv = ogr.GetDriverByName("GeoJSON")
    for name, (feats, fields) in layers.items():
        p = out_dir / f"{name}.geojson"
        if p.exists():
            p.unlink()
        gds = gdrv.CreateDataSource(str(p))
        lyr = gds.CreateLayer(name, wgs, ogr.wkbPolygon)
        for fname, ftype in fields.items():
            lyr.CreateField(ogr.FieldDefn(fname, ftype))
        for geom, attrs in feats:
            f = ogr.Feature(lyr.GetLayerDefn())
            for k, v in attrs.items():
                f.SetField(k, v)
            g = geom.Clone()
            g.Transform(tx)
            f.SetGeometry(g)
            lyr.CreateFeature(f)
            f = None
        gds = None
        print(f"wrote {p}  (EPSG:4326)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "analysis" / "shapes"))
    a = ap.parse_args()
    build(Path(a.out_dir))


if __name__ == "__main__":
    main()
