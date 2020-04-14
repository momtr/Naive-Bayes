// Naive Bayes
// Moritz Mitterdorfer, 2020

/**
 * Implementation of the  Naive Bayes Classifier
 */
class NaiveBayesClassifier {

    /**
     * Constructor
     * Data Object must have:
     *  - data.classes: string array of all classes
     *  - data.features: string array of all feature names
     *  - data.training_data: array containing objects:
     *      - object.class: class name as string
     *      - object.feature_1: value_1 (number)
     * @param {object} data - data object
     */
    constructor(data) {
        this.classes = data.classes;
        this.features = data.features;
        this.data = data.training_data;
        this.formatted = {};
        for (let i of this.classes) {
            // has to be done each time, otherwise it would create references!
            let obj = {};
            for (let j of this.features) {
                obj[j] = [];
            }
            this.formatted[i] = obj;
        }
        for (let i of this.data) {
            for (let j of this.features) {
                this.formatted[i.class][j].push(i[j]);
            }
        }
        // n is the number of all training examples
        this.n = data.training_data.length;
        // this object stores the number of training examples belonging to a certain class
        this.countClasses = {};
        for (let i of this.classes) {
            // TODO: now, we are assuming no feature is undefined!
            this.countClasses[i] = this.formatted[i][this.features[0]].length;
        }

        // training
        // (1) P(Class) = N(all) / count(Class)
        // the propability for each class is claculated and stored in an object
        this.propabilities = {};
        // (2) P(Feature|Class) -> distribution
        // the propability for a feature belonging to a certain class is calculated using the gaussian bell curve
        // for that we need the mean and the standard deviation
        this.statistics = {};
        for (let i of this.classes) {
            // set one
            this.propabilities[i] = this.countClasses[i] / this.n;
            // set two
            this.statistics[i] = {};
            for (let j of this.features) {
                this.statistics[i][j] = {};
                let m = this.mean(this.formatted[i][j]);
                let sd = this.standardDeviation(this.formatted[i][j], m)
                this.statistics[i][j]["mean"] = m;
                this.statistics[i][j]["sd"] = sd;
            }
        }
    }

    /**
     * Returns the mean of an array
     * @param {array} array - array containing numbers 
     */
    mean(array) {
        let sum = 0;
        for (let i of array) {
            sum += i;
        }
        return sum / array.length;
    }

    /**
     * returns the standard deviation of an array
     * @param {array} array -  array containing numbers
     * @param {number} m - mean can be specified, it not, it is calculated
     */
    standardDeviation(array, m) {
        if (!m)
            m = this.mean(array);
        let n = array.length;
        let sum = 0;
        for (let i of array) {
            sum += (i - m) * (i - m);
        }
        return Math.pow((sum / n), (1 / 2));
    }

    /**
     * Returns the propability for a feature being part of a class
     * @param {object} feature_object - object having all features stored: object.feature_1 = value_1
     */
    predict(feature_object) {
        // class array
        let ret = [];
        // go through all classes
        for (let i of this.classes) {
            // P = P(Class) * P(Feature|Class) = 
            //   = P(Class) * P(F_1|Class) * P(F_n|Class)
            // get the prop. for that class
            let p = this.propabilities[i];
            // go through all features
            for (let j of this.features) {
                // calculate prop. with gaussian bell curve
                let sd = this.statistics[i][j].sd;
                let gauss_1 = 1 / (sd * Math.pow(2 * Math.PI, (1 / 2)));
                let x = feature_object[j];
                let m = this.statistics[i][j].mean;
                let gauss_2 = Math.exp(-(1 / 2) * Math.pow((x - m) / sd, 2));
                p *= (gauss_1 * gauss_2);
            }
            // add class and propability to array
            ret.push({
                class: i,
                propability: p
            });
        }
        // sort the array (max: array[0])
        ret.sort((a, b) => {
            if (a.propability < b.propability) return 1;
            else return -1;
        });
        // finally, return the array containing classes and their propabilities
        return ret;
    }

    /**
     * returns the score of the model (0 <= score <= 1)
     * @param {array} data - data array of data objects
     * [ {x: [..], y: .. }, .., {..} ]
     */
    score(data) {
        let n = data.length; 
        let right = 0;
        for(let i of data) {
            let featureObj = {};
            for(let f = 0; f < i.x.length; f++) {
                let feature = this.features[f];
                featureObj[feature] = i.x[f];
            }
            let prediction = this.predict(featureObj)[0].class
            if(prediction == i.y)
                right++;
        }
        return right / n;
    }

}