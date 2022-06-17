// Test import of a JavaScript module
import { example } from '@/js/example'

// Test import of an asset
import webpackLogo from '@/images/webpack-logo.svg'

import startImage from '@/images/monkey.png'

import * as tf from '@tensorflow/tfjs';

// Test import of styles
import '@/styles/index.scss'

// Appending to the DOM
const logo = document.createElement('img')
logo.src = startImage

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
app.append(logo, heading, startButton )


async function main() {
    let model = await tf.loadGraphModel("https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/feature_vector/3/default/1", { fromTFHub: true })
    // Computation
    // f(a, b) = a * b
    // const f = (a, b) => {
    //     let x = a.matMul(b);
    //     x = tf.relu(x);
    //     return x;
    // }
    const f = (a) => {
        let x = model.execute(a);
        return x
    }
    
    // df/da = b, df/db = a
    const g = tf.valueAndGrads(f);
    
    const a = tf.randomNormal([1,224,224,3]);
    // const b = tf.randomNormal([3,2]);
    const {value, grads} = g([a]);
    
    const [da] = grads;
    
    console.log('value');
    value.print();
    
    console.log('da');
    da.print(true);
    // console.log('db');
    // db.print();
}

main()