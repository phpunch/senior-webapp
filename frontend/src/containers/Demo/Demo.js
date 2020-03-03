import React, { Component } from "react";
import ReactYoutube from "../../components/ReactYoutube";

import Papa from "papaparse";
import csv from "../Politician/politician_list.csv";

import {
  Button,
  CircularProgress,
  Grid,
  Container,
  Box
} from "@material-ui/core";
import TextInput from "../../components/TextInput";

import { makeStyles } from "@material-ui/styles";

class Demo extends Component {
  state = {
    currentTime: 0,
    politician_idx2spk: [],
    politician_spk2idx: {},
    prediction_original: null,
    prediction: null,
    value: ""
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

  setPrediction = prediction => {
    this.setState(
      {
        prediction: prediction,
        prediction_original: prediction
      },
      () => {
        console.log(this.state.prediction);
      }
    );
  };

  readFile = () => {
    const fileUrl = this.props.prediction_filename; // "result_prod_c2_v2_clean.txt"; // provide file location in public folder

    fetch(fileUrl)
      .then(r => r.text())
      .then(text => text.split("\n"))
      .then(this.setPrediction);
  };

  setData = data => {
    const politician_idx2spk = [];
    data.data.map(element => {
      return politician_idx2spk.push(element.name);
    });

    const politician_spk2idx = {};
    for (let i = 0; i <= politician_idx2spk.length; i++) {
      politician_spk2idx[politician_idx2spk[i]] = i;
    }
    console.log(politician_spk2idx);

    this.setState(
      {
        politician_idx2spk: politician_idx2spk,
        politician_spk2idx: politician_spk2idx
      },
      () => {
        console.log(this.state.politician_idx2spk);
      }
    );
  };

  readCSV = () => {
    Papa.parse(csv, {
      download: true,
      header: true,
      complete: this.setData
    });
  };

  setValue = (event, value) => {
    const prediction = [...this.state.prediction]; // copy prediction
    const idx = Math.floor(this.state.currentTime / 3); // calculate index by current time

    const new_label = this.state.politician_spk2idx[value]; // get label from dictation
    prediction[idx] = new_label; // put label back in prediction

    this.setState({
      value: value,
      prediction: prediction
    });
  };

  render() {
    let originalPredict = "";
    let textInput = <CircularProgress />;
    if (this.state.prediction && this.state.currentTime) {
      const idx = Math.floor(this.state.currentTime / 3);
      let label = parseInt(this.state.prediction[idx]);

      textInput = (
        <TextInput
          options={this.state.politician_idx2spk}
          onInputChange={this.setValue}
          defaultValue={this.state.politician_idx2spk[label]}
          key={`movingContactName:${this.state.politician_idx2spk[label]}`}
        />
      );
    }
    if (this.state.prediction_original && this.state.currentTime) {
      const idx = Math.floor(this.state.currentTime / 3);
      let label = parseInt(this.state.prediction_original[idx]);

      originalPredict = this.state.politician_idx2spk[label];
    }

    return (
      <Grid container spacing={4} justify="center">
        <Grid item lg={6} sm={12} xl={12} xs={12} >
          <Box mx="auto" bgcolor="background.paper" p={5}>
            <ReactYoutube
              mx="auto"
              videoId={this.props.videoId}
              setCurrentTimeHandler={this.setCurrentTimeHandler}
            />
          </Box>
        </Grid>
        <Grid item lg={3} sm={6} xl={3} xs={12} justify="center">
          <div className="card card-body">
            Current Time: {this.state.currentTime}
          </div>
          <div className="card card-body">
            Original Predict: {originalPredict}
          </div>
          <div className="card card-body">
            Predict:
            <div className="card card-body">{textInput}</div>
          </div>
          <Button variant="contained" color="primary">
            Print!
          </Button>
        </Grid>
      </Grid>
    );
  }
}

export default Demo;
