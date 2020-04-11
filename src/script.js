let data = {};
let classifier;
let currentClass = 'a';

function setup() {

    data.classes = ["a", "b", "c"];
    data.features = ["x", "y"];
    data.training_data = [];

    // DOM
    createCanvas(600, 600);
    noStroke();
    
    
}

function draw() {

    background(40);

    if(classifier) {
        for(let y = 0; y < height; y += 20) {
            for(let x = 0; x < width; x += 20) {
                let result = classifier.predict({ x, y });
                let label = result[0].class;
                if(label == 'a')        fill(255, 0, 0, 50);
                else if(label == 'b')   fill(0, 255, 0, 50);
                else if(label == 'c')   fill(0, 0, 255, 50);
                rect(x, y, 20, 20);
            }
        }
    }

    for(let i of data.training_data) {
        if(i.class == 'a')      fill(255, 0, 0);
        else if(i.class == 'b') fill(0, 255, 0);
        else if(i.class == 'c') fill(0, 0, 255);
        ellipse(i.x, i.y, 30, 30);
    }

}

function keyPressed() {
    if(key == 'a' || key == 'b' || key == 'c') {
        currentClass = key;
    }
    else if(key == 's') {
        classifier = new NaiveBayes(data);
    }
}

function mousePressed() {
    data.training_data.push({
        class: currentClass,
        x: mouseX,
        y: mouseY
    });
}