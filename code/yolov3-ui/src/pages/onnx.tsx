
import React from 'react'
import {connect} from 'dva'
import {Row, Col, Form, Select, Button,Upload,Icon,message, Collapse} from 'antd'
import {InferenceSession, Tensor} from 'onnxjs'
import loadImage from 'blueimp-load-image'
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import {yolo, yoloTransforms} from '../utils/index';
import classNames from '../data/yolo_classes';

import styles from './onnx.css'

const { Option } = Select;


@connect(({ onnx }) => {
  return {
    onnx
  }
})
export default class onnx extends React.Component {

  state = {
    curSession:null,
    curModel:'yolov3',
    imgsize:{
      width:0,
      height:0,
    },
    canvasSize:{
      x:0,
      y:0,
      w:0,
      h:0,
    }
  }

  /*后端改变效应事件*/
  backendChangeHandler = (value: any) =>{
    // const {sessionList} = this.state
    // const session = sessionList.filter((item) => {return item.key === value})[0]
    // this.setState({...this.state,curSession:session.value})
  }

  /*模型改变效应事件*/
  modelChangeHandler = (value: any) =>{
    const { dispatch } = this.props
    const myOnnxSession = new InferenceSession()
    try {
        dispatch({type:'onnx/getModel',payload:{modelName:value}})
        .then((response) => {
            response.arrayBuffer().then((modelfile) => {
              myOnnxSession.loadModel(modelfile)
              this.setState({...this.state,curSession:myOnnxSession,curModel:value})
           })
         })
    } catch (e){
      throw new Error('Error: Backend not supported. ');
    }
  }

  /* 选择图片处理事件 */
  selecedImgHandler = (e: any) =>{
    if (!e.target.files[0]) {
      const element = document.getElementById('input-canvas') as HTMLCanvasElement;
      const ctx = element.getContext('2d')as CanvasRenderingContext2D;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      return;
    }
    loadImage(
      e.target.files[0],
      img => {
        this.clearRects()
        const element = document.getElementById('input-canvas') as HTMLCanvasElement
        const ctx = element.getContext('2d') as CanvasRenderingContext2D
        const imageWidth = (img as HTMLImageElement).width
        const imageHeight = (img as HTMLImageElement).height

        const max_height = imageWidth > imageHeight? imageWidth:imageHeight
        const gain = 416/max_height
        const newWidth = imageWidth*gain
        const newHeight = imageHeight*gain

        let x1 = 0
        let y1 = 0

        if(newWidth<416){
          x1 = (416 - newWidth)/2
        }
        if(newHeight<416) {
          y1 = (416 - newHeight)/2
        }

        ctx.drawImage(img as HTMLImageElement, 0, 0, imageWidth, imageHeight, x1, y1, newWidth, newHeight)
        this.setState({...this.state, imgsize:{width:imageWidth,height:imageHeight},canvasSize:{x:x1,y:y1,w:newWidth,h:newHeight} })
        
        this.drowImg(img,'input-canvas-orgi')
        
        this.runModel(ctx)
      },
      {
        cover: true,
        crop: true,
        canvas: true,
        crossOrigin: 'Anonymous',
      }
    )
  }

  drowImg = (img:HTMLImageElement,elem_id:string) =>{
    const element = document.getElementById(elem_id) as HTMLCanvasElement
    const ctx = element.getContext('2d') as CanvasRenderingContext2D
    ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, element.width, element.height)
  }



  /* 运行模型 */
  runModel = (ctx:any) => {
    const { curSession } = this.state
    const data = this.preprocess(ctx)
    curSession.run([data]).then((result) => {
        const outputdata = result.values()
        const scores = outputdata.next().value
        const boxs = outputdata.next().value
        this.postprocessyolov3(scores,boxs)
    })
  }

  postprocessyolov3(scores: Tensor, boxs: Tensor) {
    const max = yoloTransforms.max2(scores,1)
    const max_score = max[0]
    const zipped = []
    for(let i=0;i<max[0].size;i++){
      if(max[0].data[i]>0.5){
        const x = boxs.data[i*4+0]
        const y = boxs.data[i*4+1]
        const w = boxs.data[i*4+2]
        const h = boxs.data[i*4+3]
        const x1 = x - w/2
        const x2 = x + w/3
        const y1 = y - h/2
        const y2 = y + h/2
        zipped.push([max[0].data[i],max[1].data[i],[x1,y1,x2,y2]])
      }
    }
    const sortedBoxes = zipped.sort((a: number[], b: number[]) => b[0] - a[0])

    const selectedBoxes: any[] = [];

    sortedBoxes.forEach((box: any[]) => {
      let add = true;
      for (let i=0; i < selectedBoxes.length; i++) {
        const curIou = this.box_iou(box[2], selectedBoxes[i][2]);
        if (curIou > 0.5) {
          add = false;
          break;
        }
      }
      if (add) {
        selectedBoxes.push(box);
      }
    })

    selectedBoxes.forEach((box) => {
      const x1y1x2y2 = this.scale_box(box[2])
      this.drawRect(x1y1x2y2[0], x1y1x2y2[1], (x1y1x2y2[2]-x1y1x2y2[0]), (x1y1x2y2[3]-x1y1x2y2[1]),`${classNames[box[1]]} Confidence: ${Math.round(box[0] * 100)}%`);
    })
  }

  scale_box = (x1y1x2y2:any[]) =>{
    const {imgsize:{width,height}} = this.state

    let x1 = x1y1x2y2[0]*416
    let y1 = x1y1x2y2[1]*416
    let x2 = x1y1x2y2[2]*416
    let y2 = x1y1x2y2[3]*416

    return [x1,y1,x2,y2]
  }

  box_intersection = (a: number[], b: number[]) => {
    const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
    const h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    if (w < 0 || h < 0) {
      return 0;
    }
    return w * h;
  }
  
  box_union = (a: number[], b: number[]) =>  {
    const i = this.box_intersection(a, b);
    return (a[3] - a[1]) * (a[2] - a[0]) + (b[3] - b[1]) * (b[2] - b[0]) - i;
  }
  
  box_iou = (a: number[], b: number[]) => {
    return this.box_intersection(a, b) / this.box_union(a, b);
  }

  /* 图片数据预处理 */
  preprocess(ctx: CanvasRenderingContext2D): Tensor {
    const { canvasSize:{x,y,w,h} } = this.state
    const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const { data, width, height } = imageData;
    // data processing
    const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [1, 3, width, height]);

    ops.assign(dataProcessedTensor.pick(0, 0, null, null), dataTensor.pick(null, null, 0));
    ops.assign(dataProcessedTensor.pick(0, 1, null, null), dataTensor.pick(null, null, 1));
    ops.assign(dataProcessedTensor.pick(0, 2, null, null), dataTensor.pick(null, null, 2));
    
    const tensor = new Tensor(new Float32Array(width* height* 3), 'float32', [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataProcessedTensor.data);
    const two = new Tensor([255.0], 'float32');
    const newTensor = yoloTransforms.div(tensor, two)
    return newTensor;
  }

  /** yolov2后处理 */
  postprocess(tensor: Tensor) {
    const originalOutput = new Tensor(tensor.data as Float32Array, 'float32', [1, 125, 13, 13])
    const outputTensor = yoloTransforms.transpose(originalOutput, [0, 2, 3, 1])
    yolo.postprocess(outputTensor, 20).then((boxes) => {
      boxes.forEach((box) => {
        const {
          top, left, bottom, right, classProb, className,
        } = box;
        this.drawRect(left, top, right-left, bottom-top,`${className} Confidence: ${Math.round(classProb * 100)}%`);
      })
    })
  }

  /** 画框 */
  drawRect(x: number, y: number, w: number, h: number, text = '', color = 'red') {
    console.log('xywh',x,y,w,h)
    const rect = document.createElement('div');
    rect.style.cssText = `top: ${y}px; left: ${x}px; width: ${w}px; height: ${h}px; border-color: ${color};position:absolute;border:1px solid red;`
    const label = document.createElement('div');
    label.style.cssText = `background: white;color: black;opacity: 0.8;font-size: 12px;padding: 3px;text-transform: capitalize;white-space: nowrap;`
    label.innerText = text;
    rect.appendChild(label);
    (document.getElementById('webcam-container') as HTMLElement).appendChild(rect);
  }

  /** 画框 */
  drawRectOrgi(x: number, y: number, w: number, h: number, text = '', color = 'red') {
    const rect = document.createElement('div');
    rect.style.cssText = `top: ${y}px; left: ${x}px; width: ${w}px; height: ${h}px; border-color: ${color};position: absolute;border: 1px solid red;`
    const label = document.createElement('div');
    label.style.cssText = `background: white;color: black;opacity: 0.8;font-size: 12px;padding: 3px;text-transform: capitalize;white-space: nowrap;`
    label.innerText = text;
    rect.appendChild(label);
    (document.getElementById('webcam-container-orgi') as HTMLElement).appendChild(rect);
  }

  /** 清除框 */
  clearRects() {
    const element = document.getElementById('webcam-container') as HTMLCanvasElement;
    while (element.childNodes.length > 1)  {
      element.removeChild(element.childNodes[1]);
    }
  }

  
  render() {
    const { onnx }  = this.props
    return (
      <div>
        <Row>
          <Col span={8}>
            <Row style={{marginTop:10}}>
              <Col span={4} style={{textAlign:"right", marginTop:5}}>后端:</Col>
              <Col span={10} style={{textAlign:"left", marginLeft:10}}>
                <Select onChange={this.backendChangeHandler} style={{ width: 200}} >
                  <Option value="wasm">CPU-WebAssembly</Option>
                  <Option value="webgl">GPU-WebGL</Option>
                </Select>
              </Col>
            </Row>
            <Row style={{marginTop:10}}>
              <Col span={4} style={{textAlign:"right", marginTop:5}}>模型:</Col>
              <Col span={10} style={{textAlign:"left", marginLeft:10}}>
                <Select onChange={this.modelChangeHandler} style={{ width: 200}}>
                  <Option value="yolov3">yolov3</Option>
                </Select>
              </Col>
            </Row>
            <Row style={{marginTop:10}}>
              <label className={styles.inputs}>
                上传图片
                <input style={{display: "none"}}  type="file" onChange={this.selecedImgHandler}/>
              </label>
            </Row>
          </Col>
          <Col span={16}>
            <div className={styles.webcamcontainer} id="webcam-container">
              <canvas id="input-canvas"  width={416} height={416}/>
            </div>
          </Col>
        </Row>
        <Row>
          <div style={{width:this.state.imgsize.width,height:this.state.imgsize.height}} id="webcam-container-orgi">
            <canvas id="input-canvas-orgi"  width={this.state.imgsize.width} height={this.state.imgsize.height}/>
          </div>
        </Row>
      </div>
    )
  }
}
