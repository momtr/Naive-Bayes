// object storing all classes, features and training examples
let data = {};
// varaible storing the Naive Bayes Classifier Instance
let classifier;
// class 'a' is pre-selected
let currentClass = 'a';

// called at the beginning
function setup() {
    // specifie classes (a, b, c)
    data.classes = ["a", "b", "c"];
    // specifie features (x-, y-coordinates)
    data.features = ["x", "y"];
    data.training_data = [];
    // DOM
    createCanvas(600, 600);
    noStroke();
}

// called every tick
function draw() {
    // set background color to rgb(40,40,40)
    background(40);

    // if the classifier instance has been created
    if (classifier) {
        // go through all 20x20 squares of the canvas
        for (let y = 0; y < height; y += 20) {
            for (let x = 0; x < width; x += 20) {
                // predict class using the Naive Bayes Classifier instance
                let result = classifier.predict({ x, y });
                // get the class name
                let label = result[0].class;
                // specifie color according to class 
                if (label == 'a') fill(255, 0, 0, 50);
                else if (label == 'b') fill(0, 255, 0, 50);
                else if (label == 'c') fill(0, 0, 255, 50);
                // draw a rect
                rect(x, y, 20, 20);
            }
        }
    }

    // draw all points that have been added as training examples
    for (let i of data.training_data) {
        if (i.class == 'a') fill(255, 0, 0);
        else if (i.class == 'b') fill(0, 255, 0);
        else if (i.class == 'c') fill(0, 0, 255);
        ellipse(i.x, i.y, 30, 30);
    }

}

// if a key has been pressed
function keyPressed() {
    if (key == 'a' || key == 'b' || key == 'c') {
        // change the current class to the key's value
        currentClass = key;
    }
    else if (key == 's') {
        // if 's' has been pressed, create a Naive Bayes Classifier instance
        classifier = new NaiveBayesClassifier(data);
    }
}

// if the mouse has been pressed, add a new training example 
function mousePressed() {
    data.training_data.push({
        class: currentClass,
        x: mouseX,
        y: mouseY
    });
}