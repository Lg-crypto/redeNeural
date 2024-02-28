const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
    const sigmoidX = sigmoid(x);
    return sigmoidX * (1 - sigmoidX);
}

class NeuralNetwork {
    constructor() {
        this.weights = [Math.random(), Math.random()];
        this.bias = Math.random();
        this.learningRate = 0.1;
    }

    predict(input1, input2) {
        const weightedSum = input1 * this.weights[0] + input2 * this.weights[1] + this.bias;
        return sigmoid(weightedSum);
    }

    train(input1, input2, target) {
        const output = this.predict(input1, input2);
        const error = target - output;
        const dError = error * sigmoidDerivative(output);
        const dWeights = [input1 * dError, input2 * dError];
        const dBias = dError;

        this.weights[0] += this.learningRate * dWeights[0];
        this.weights[1] += this.learningRate * dWeights[1];
        this.bias += this.learningRate * dBias;

        return error;
    }
}

const neuralNetwork = new NeuralNetwork();
const input1 = 0.5;
const input2 = 0.8;
const target = 1;
const tolerance = 0.001;

let bestError = Infinity;
let bestOutput;

function drawBall(x, y, color) {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.closePath();
}

function drawGraph() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let error = Infinity;
    let iterations = 0;
    let xPos = 0;

    while (Math.abs(error) > tolerance && xPos < canvas.width) {
        error = neuralNetwork.train(input1, input2, target);
        iterations++;
        const output = neuralNetwork.predict(input1, input2);
        const yPos = output * canvas.height;
        drawBall(xPos, yPos, 'blue');

        if (Math.abs(error) < Math.abs(bestError)) {
            bestError = error;
            bestOutput = output;
        }

        xPos += 0.1;
    }

    const bestYPos = bestOutput * canvas.height;
    drawBall(xPos, bestYPos, 'yellow');
    console.log('Resultado após', iterations, 'iterações:');
    console.log('Saída:', bestOutput);
}

drawGraph();

