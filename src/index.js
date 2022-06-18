// Test import of a JavaScript module
import { example } from '@/js/example'

// Test import of an asset
import webpackLogo from '@/images/webpack-logo.svg'

import startImage from '@/images/monkey.png'

import * as tf from '@tensorflow/tfjs';

// import modelPath from './js/models/deep_dream/model.json';

// Test import of styles
import '@/styles/index.scss'

// Appending to the DOM
const canvas = document.createElement('canvas');
let img = document.createElement('img');
let ctx = canvas.getContext('2d');
// const heading = document.createElement('h1')
// heading.textContent = example()
// heading.textContent = "Deep Dream"

// const startButton = document.createElement('button');
// startButton.textContent = 'Start Deep Dream'

// Test a background image url in CSS
// const imageBackground = document.createElement('div')
// imageBackground.classList.add('image')

// Test a public folder asset
const imagePublic = document.createElement('img')
imagePublic.src = '/assets/example.png'


const app = document.querySelector('#root')
app.append(canvas) //heading, )//startButton)


async function getGradient(model, image, feature_layers=[0,1]) {
  let [b, h, w, c] = image.shape;
  image = tf.image.resizeBilinear(image, [224, 224])
  let grads = [];
  for (const layer of feature_layers) {
    const f = (a) => {
      let out = model.predict(a);
      out = out[layer]; // TODO: Very shoddy gotta change this
      return out
    }

    const g = tf.grads(f);

    // const { value, grads } = g([image]);
    const [dimage] = g([image]);
    dimage = tf.image.resizeBilinear(dimage, [h, w]);
    grads.push(dimage);
  }
  return grads;
}


const loadImageElement = path => {
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
async function loadImage() {
  let htmlImage = await loadImageElement(startImage);
  var height = htmlImage.height;
  var width = htmlImage.width;
  canvas.height = height
  canvas.width = width
  ctx.drawImage(htmlImage, 0, 0)
  let image = await tf.browser.fromPixels(canvas);

  return image;
}

async function preprocessImage(image) {
  // let {mean,variance} = tf.moments(image);
  // let std = variance.sqrt();
  // image = image.sub(mean).div(std);
  image = tf.expandDims(image, [0]);
  image = image.div(127.5).sub(1); // -1 to 1
  //  // console.log('Printing image stats: ');
  // console.log('MAX and MIN and MEAN and VAR');
  //  // tf.max(image).print()
  //  // tf.min(image).print();
  // ({mean,variance} = tf.moments(image));
  //  // mean.print();
  //  // variance.print();
  // console.log('IMAGE');
  //  // image.print(true);
  return image;
}

async function deprocessImage(image) {
  //  image.print(true)
  image = tf.cast(image.clipByValue(-1, 1).add(1)
    .mul(127.5).squeeze(), 'int32');
  return image;
}

async function updateImage(image, grads, step_size) {
  step_size /= grads.length;
  for (const grad of grads) {
    let { mean, variance } = tf.moments(grad);
    let std = variance.sqrt();
    // std.print(true)
    // grad = grad.sub(mean)
    image = image.add(grad.mul(step_size))
    //  await grad.print(true);
    //  await image.print(true)
    // console.log('HERE');
    // ({mean,variance} = tf.moments(image))
    //  // mean.print(true)
    //  // image.max().print(true)
    //  // image.min().print(true)
    image = image.clipByValue(-1, 1);
  }
  return image;
}

async function updateCanvas(image, canvas) {
  await tf.browser.toPixels(image, canvas)
}

async function main() {
  let origImage = await loadImage();
  let modelPath = '/models/deep_dream/model.json'
  let model = await tf.loadLayersModel(modelPath);
  // model.summary();


  let learning_rate = .02;
  let prepImage = await preprocessImage(origImage);
  let steps = 30;

  for (let i = 0; i < steps; i++) {
    console.log(`${i}/${steps}...`);
    let grads = await getGradient(model, prepImage, [3,4]);
    prepImage = await updateImage(prepImage, grads, learning_rate);
    let image = await deprocessImage(prepImage);
    await updateCanvas(image, canvas);
  }

  // let out = await model.predict(tf.randomNormal([1,224,224,3]))
  // // out.print(true);
  // console.log(out);
}

 main()