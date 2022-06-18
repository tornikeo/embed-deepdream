// Test import of a JavaScript module
import { example } from '@/js/example'

// Test import of an asset
import webpackLogo from '@/images/webpack-logo.svg'

import startImage from '@/images/monkey.png'

import * as tf from '@tensorflow/tfjs';

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


async function getGradient(model, image, seed) {
  let [b, h, w, c] = image.shape;
  image = tf.image.resizeBilinear(image, [224, 224])

  const f = (a) => {
    let x = model.execute(a).slice([0,42],[-1,100]);
    // x.print(true)
    //TODO: Either change the base model
    //      Or make x slice per-session fixed random
    return x
  }

  const g = tf.valueAndGrads(f);

  const { value, grads } = g([image]);

  let [dimage] = grads;
  // console.log('OUTPUT VALS');
  //  // value.print(true);

  // console.log('OUTPUT IMAGE GRAD');
  //  // dimage.print(true);

  dimage = tf.image.resizeBilinear(dimage, [h, w]);
  return dimage;
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

async function updateImage(image, grad, step_size) {
  let { mean, variance } = tf.moments(grad);
  let std = variance.sqrt();
  std.print(true)
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
  return image;
}

async function updateCanvas(image, canvas) {
  await tf.browser.toPixels(image, canvas)
}

async function main() {
  let origImage = await loadImage();
  let model = await tf.loadGraphModel("https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/2/default/1", { fromTFHub: true })

  let learning_rate = .5;
  let prepImage = await preprocessImage(origImage);
  let steps = 100;

  // canvas.addEventListener('click', async () => {
  //   // for (let i = 0; i < steps; i++) {
  //     // console.log(`${i}/${steps}...`);
  //     let grad = await getGradient(model, prepImage);
  //     prepImage = await updateImage(prepImage, grad, learning_rate);
  //     let image = await deprocessImage(prepImage);
  //     await updateCanvas(image, canvas);
  //   // }
  // })
  for (let i = 0; i < steps; i++) {
    console.log(`${i}/${steps}...`);
    let grad = await getGradient(model, prepImage);
    prepImage = await updateImage(prepImage, grad, learning_rate);
    let image = await deprocessImage(prepImage);
    await updateCanvas(image, canvas);
  }
}

main()