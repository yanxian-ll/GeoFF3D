# UAV-SLAM Scene Image Counts

- Source: `bash_scripts/benchmark/uav_slam/default_scenes.yaml`
- Count rule: files in `images/` with common image extensions; fallback to `rgb/` if `images/` is missing.
- `selected_after_stride` uses dataset/scene stride when present.

## Summary

| dataset | scenes | images | selected_after_stride | missing |
|---|---:|---:|---:|---:|
| enrich | 2 | 99 | 0 | 0 |
| npu_dronemap | 12 | 5276 | 0 | 0 |
| uavff3d_real | 11 | 10206 | 0 | 0 |
| uavscenes | 20 | 24126 | 8047 | 0 |
| urbanscene3d | 2 | 1087 | 0 | 0 |
| usegeo | 3 | 828 | 0 | 0 |

## Scenes

| dataset | scene | images | stride | selected_after_stride | status |
|---|---|---:|---:|---:|---|
| uavscenes | interval5_AMtown01 | 2589 | 3 | 863 | ok |
| uavscenes | interval5_AMtown02 | 1380 | 3 | 460 | ok |
| uavscenes | interval5_AMtown03 | 1120 | 3 | 374 | ok |
| uavscenes | interval5_AMvalley01 | 2260 | 3 | 754 | ok |
| uavscenes | interval5_AMvalley02 | 1266 | 3 | 422 | ok |
| uavscenes | interval5_AMvalley03 | 961 | 3 | 321 | ok |
| uavscenes | interval5_HKairport_GNSS_Evening | 1415 | 3 | 472 | ok |
| uavscenes | interval5_HKairport_GNSS01 | 1356 | 3 | 452 | ok |
| uavscenes | interval5_HKairport_GNSS02 | 747 | 3 | 249 | ok |
| uavscenes | interval5_HKairport_GNSS03 | 630 | 3 | 210 | ok |
| uavscenes | interval5_HKairport01 | 1440 | 3 | 480 | ok |
| uavscenes | interval5_HKairport02 | 786 | 3 | 262 | ok |
| uavscenes | interval5_HKairport03 | 605 | 3 | 202 | ok |
| uavscenes | interval5_HKisland_GNSS_Evening | 2230 | 3 | 744 | ok |
| uavscenes | interval5_HKisland_GNSS01 | 1356 | 3 | 452 | ok |
| uavscenes | interval5_HKisland_GNSS02 | 760 | 3 | 254 | ok |
| uavscenes | interval5_HKisland_GNSS03 | 534 | 3 | 178 | ok |
| uavscenes | interval5_HKisland01 | 1317 | 3 | 439 | ok |
| uavscenes | interval5_HKisland02 | 775 | 3 | 259 | ok |
| uavscenes | interval5_HKisland03 | 599 | 3 | 200 | ok |
| usegeo | dataset1 | 224 |  |  | ok |
| usegeo | dataset2 | 327 |  |  | ok |
| usegeo | dataset3 | 277 |  |  | ok |
| uavff3d_real | nanfang_ndir2 | 579 |  |  | ok |
| uavff3d_real | yanghaitang_ndir2 | 387 |  |  | ok |
| uavff3d_real | xiaoxiang_ndir2 | 1004 |  |  | ok |
| uavff3d_real | nanfang_part0_ndir | 1177 |  |  | ok |
| uavff3d_real | nanfang_part1_ndir | 1165 |  |  | ok |
| uavff3d_real | xiaoxiang_part0_ndir | 989 |  |  | ok |
| uavff3d_real | xiaoxiang_part1_ndir | 986 |  |  | ok |
| uavff3d_real | xiaoxiang_part2_ndir | 1014 |  |  | ok |
| uavff3d_real | xiaoxiang_part3_ndir | 1007 |  |  | ok |
| uavff3d_real | yanghaitang_part0_ndir | 929 |  |  | ok |
| uavff3d_real | yanghaitang_part1_ndir | 969 |  |  | ok |
| enrich | aerial_ndiir2 | 39 |  |  | ok |
| enrich | aerial_ndir | 60 |  |  | ok |
| npu_dronemap | gopro-monticules-kfs | 395 |  |  | ok |
| npu_dronemap | gopro-npu-kfs | 337 |  |  | ok |
| npu_dronemap | gopro-saplings-kfs | 482 |  |  | ok |
| npu_dronemap | phantom3-factory-kfs | 402 |  |  | ok |
| npu_dronemap | phantom3-freeway-kfs | 415 |  |  | ok |
| npu_dronemap | phantom3-grass-kfs | 648 |  |  | ok |
| npu_dronemap | phantom3-highflower-kfs | 285 |  |  | ok |
| npu_dronemap | phantom3-huangqi-kfs | 393 |  |  | ok |
| npu_dronemap | phantom3-ieu-kfs | 467 |  |  | ok |
| npu_dronemap | phantom3-lowlower-kfs | 589 |  |  | ok |
| npu_dronemap | phantom3-npu-kfs | 457 |  |  | ok |
| npu_dronemap | phantom3-village-kfs | 406 |  |  | ok |
| urbanscene3d | artsci_ndir | 594 |  |  | ok |
| urbanscene3d | polytech_ndir | 493 |  |  | ok |
