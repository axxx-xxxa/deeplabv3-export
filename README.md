## **Deeplabv3 export**

| argument |          help           | request  | type |  default  |
| :------: | :---------------------: | :------: | :--: | :-------: |
|   app    | pth2onnx or onnx2engine |   must   | str  |     -     |
|    -n    |       num classes       | optional | int  |     3     |
|    -b    |        backbone         | optional | str  | mobilenet |
|    -d    |    downsample_factor    | optional | int  |    16     |
|    -p    |  pth_path or onnx_path  |   must   | str  |     -     |

## **Quick Start Examples**

```
pth2onnx:
python export_plus.py pth2onnx -p mobilenetv2.pth (-n -b -d)

output:mobilenetv2.onnx in current folder

onnx2enging:
python export_plus.py onnx2enging -p mobilenetv2.onnx

output:mobilenetv2.engine in current folder

```

```
!!! nets folder should be in the current folder
```

