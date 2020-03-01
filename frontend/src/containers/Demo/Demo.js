import React, { Component } from "react";
import ReactYoutube from "../../components/ReactYoutube";

import Papa from "papaparse";
import csv from "../Politician/politician_list.csv";

class Demo extends Component {
  state = {
    currentTime: 0,
    politician_list: [],
    prediction: null
  };

  setCurrentTimeHandler = time => {
    this.setState({
      currentTime: time
    });
  };

  componentDidMount = () => {
    this.readFile();
    this.readCSV();
  };

  setPrediction = (text) => {
    this.setState({
      prediction: text
    }, () => {
      console.log(this.state.prediction)
    })
  }

  readFile = () => {
    const fileUrl = this.props.prediction_filename // "result_prod_c2_v2_clean.txt"; // provide file location in public folder

    fetch(fileUrl)
      .then(r => r.text())
      .then(text => text.split("\n"))
      .then(this.setPrediction);
  };

  setData = data => {
    this.setState({
      politician_list: data.data
    }, () => {
      console.log(this.state.politicians)
    });
  };

  readCSV = () => {
    Papa.parse(csv, {
      download: true,
      header: true,
      complete: this.setData
    });
  };

  render() {
    let name = ""
    if (this.state.prediction && this.state.currentTime) {
      const idx = Math.floor(this.state.currentTime / 3) 
      // console.log(idx)
      name = this.state.politician_list[parseInt(this.state.prediction[idx])].name
    }
    
    return (
      <div className="container">
        <div className="row">
          <div className="offset-md-3 col-md-6">
            <ReactYoutube
              // videoId="Y70hiPNxe4Q"
              // videoId="kHk5muJUwuw"
              videoId={this.props.videoId}
              setCurrentTimeHandler={this.setCurrentTimeHandler}
            />
            {/* <div className="card card-body">
              Current Time: {this.state.currentTime}
            </div> */}
            <div className="card card-body">
              Predict: {name}
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default Demo;
