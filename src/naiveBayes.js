class NaiveBayes {

    constructor(data) {
        this.classes = data.classes;
        this.features = data.features;
        this.data = data.training_data;
        this.formatted = {};
        for(let i of this.classes) {
            // has to be done each time, otherwise it would create references!
            let obj = {};
            for(let j of this.features) {
                obj[j] = [];
            }
            this.formatted[i] = obj;
        }
        for(let i of this.data) {
            for(let j of this.features) {
                this.formatted[i.class][j].push(i[j]);
            }
        }
        this.n = data.training_data.length;
        this.countClasses = {};
        for(let i of this.classes) {
            // TODO: now, we are assuming no feature is undefined!
            this.countClasses[i] = this.formatted[i][this.features[0]].length;
        }
        // train
        // (1) P(Class) = N(all) / count(Class)
        this.propabilities = {};
        // (2) P(Feature|Class) -> distribution
        this.statistics = {};
        for(let i of this.classes) {
            // 1
            this.propabilities[i] = this.countClasses[i] / this.n;
            // 2
            this.statistics[i] = {};
            for(let j of this.features) {
                this.statistics[i][j] = {};
                let m = this.mean(this.formatted[i][j]);
                let sd = this.standardDeviation(this.formatted[i][j], m)
                this.statistics[i][j]["mean"] = m;
                this.statistics[i][j]["sd"] = sd;
            }
        }
    }

    mean(array) {
        let sum = 0;
        for(let i of array) {
            sum += i;
        }
        return sum / array.length;
    }

    standardDeviation(array, m) {
        if(!m)
            m = this.mean(array);
        let n = array.length;
        let sum = 0;
        for(let i of array) {
            sum += (i - m) * (i - m);
        }
        return Math.pow((sum / n), (1/2));
    }


    predict(feature_object) {
        // we get a feature array and map it to a class
        let ret = [];
        for(let i of this.classes) {
            // P = P(Class) * P(Feature|Class) = 
            //   = P(Class) * P(F_1|Class) * P(F_n|Class)
            let p = this.propabilities[i];
            for(let j of this.features) {
                // calculate propb. with gaussian bell curve
                let sd = this.statistics[i][j].sd;
                let gauss_1 = 1 / (sd * Math.pow(2 * Math.PI, (1/2)));
                let x = feature_object[j];
                let m = this.statistics[i][j].mean;
                let gauss_2 = Math.exp(-(1/2) * Math.pow((x - m) / sd, 2));
                // console.log((gauss_1 * gauss_2));
                p *= (gauss_1 * gauss_2);
            }
            ret.push({
                class: i,
                propability: p
            });
        }
        ret.sort((a, b) => {
            if(a.propability < b.propability) return 1;
            else                              return -1;
        });
        return ret;
    }

}