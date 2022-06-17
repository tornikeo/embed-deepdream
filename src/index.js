// Test import of a JavaScript module
import { example } from '@/js/example'

// Test import of an asset
import webpackLogo from '@/images/webpack-logo.svg'

import startImage from '@/images/monkey.png'

import * as tf from '@tensorflow/tfjs';

// Test import of styles
import '@/styles/index.scss'
import { $dataMetaSchema } from 'ajv';

// Appending to the DOM

const canvas = document.createElement('canvas');
let img = document.createElement('img');
let ctx = canvas.getContext('2d');
const heading = document.createElement('h1')
// heading.textContent = example()
heading.textContent = "Deep Dream"

const startButton = document.createElement('button');
startButton.textContent = 'Start Deep Dream'

// Test a background image url in CSS
const imageBackground = document.createElement('div')
imageBackground.classList.add('image')

// Test a public folder asset
const imagePublic = document.createElement('img')
imagePublic.src = '/assets/example.png'


const app = document.querySelector('#root')
app.append(canvas, heading, startButton )

const loadImage = path => {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.crossOrigin = 'Anonymous' // to avoid CORS if used with Canvas
      img.src = path
      img.onload = () => {
        resolve(img)
      }
      img.onerror = e => {
        reject(e)
      }
    })
}

async function main() {
    let htmlImage = await loadImage(startImage);
    var height = htmlImage.height;
    var width = htmlImage.width;
    canvas.height = height
    canvas.width = width
    ctx.drawImage(htmlImage, 0, 0)

    let image = await tf.browser.fromPixels(canvas);
    let model = await tf.loadGraphModel("https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1", { fromTFHub: true })
    // Computation
    // f(a, b) = a * b
    // const f = (a, b) => {
    //     let x = a.matMul(b);
    //     x = tf.relu(x);
    //     return x;
    // }
    image = tf.expandDims(image, [0]);
    let {mean,variance} = tf.moments(image);
    let std = variance.sqrt()
    image = image.sub(mean).div(std);
    image = tf.image.resizeBilinear(image, [224,224])
    console.log('Printing image stats: ');
    console.log('MAX and MIN and MEAN and VAR');
    tf.max(image).print()
    tf.min(image).print();
    ({mean,variance} = tf.moments(image));
    mean.print();
    variance.print();
    console.log('IMAGE');
    image.print(true);


    const f = (a) => {
        let x = model.execute(a);
        return x
    }
    
    const g = tf.valueAndGrads(f);
    
    const {value, grads} = g([image]);
    
    const [dimage] = grads;    
    console.log('OUTPUT VALS');
    value.print(true);

    console.log('OUTPUT IMAGE GRAD');
    dimage.print(true);
}

main()