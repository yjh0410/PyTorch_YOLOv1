# PyTorch_YOLOv1

## Network

- Backbone: ResNet-18
- Neck: SPP

## Tricks

- [x] multi-scale

## Eval

VOC2007 test

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> mAP </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv1-320</th><td bgcolor=white> VOC2007 test </td><td bgcolor=white> 64.8 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv1-416</th><td bgcolor=white> VOC2007 test </td><td bgcolor=white> 69.2 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv1-512</th><td bgcolor=white> VOC2007 test </td><td bgcolor=white> 71.8 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv1-608</th><td bgcolor=white> VOC2007 test </td><td bgcolor=white> 73.3 </td></tr>
</table></tbody>

COCO val

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv1-320</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 13.7 </td><td bgcolor=white> 29.6 </td><td bgcolor=white> 11.3 </td><td bgcolor=white> 1.6 </td><td bgcolor=white> 11.5 </td><td bgcolor=white> 28.6 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv1-416</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 16.4 </td><td bgcolor=white> 34.7 </td><td bgcolor=white> 13.9 </td><td bgcolor=white> 3.1 </td><td bgcolor=white> 15.6 </td><td bgcolor=white> 31.9 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv1-512</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 18.1 </td><td bgcolor=white> 37.9 </td><td bgcolor=white> 15.5 </td><td bgcolor=white> 4.3 </td><td bgcolor=white> 18.5 </td><td bgcolor=white> 32.0 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> Our YOLOv1-608</th><td bgcolor=white> COCO eval </td><td bgcolor=white> 18.6 </td><td bgcolor=white> 39.0 </td><td bgcolor=white> 15.6 </td><td bgcolor=white> 5.5 </td><td bgcolor=white> 20.7 </td><td bgcolor=white> 30.6 </td></tr>
</table></tbody>
