// Test import of a JavaScript module
import { example } from '@/js/example'

// Test import of an asset
import webpackLogo from '@/images/webpack-logo.svg'

import startImage from '@/images/monkey.png'

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
