import React, { Component } from "react";
import ReactYoutube from "../../components/ReactYoutube";
import axios from "axios";
import queryString from "query-string";

import Papa from "papaparse";
import csv from "../Politician/politician_list.csv";

import { Button, CircularProgress, Grid, Box } from "@material-ui/core";
import TextInput from "../../components/TextInput";
import Card from "../../components/Card";
import Chart from '../../components/Chart'

class DemoYoutube extends Component {
  state = {
    currentTime: 0,
    youtube_url:
      "https://www.youtube.com/watch?v=kHk5muJUwuw&feature=emb_title",
    prediction: null,
    prediction_original: null,
    politician_idx2spk: [],
    politician_spk2idx: {},
    is_loading: false,
    is_saved: false
  };

  setCurrentTimeHandler = time => {
    this.setState({
      currentTime: time
    });
  };

  textInputHandler = event => {
    this.setState({
      youtube_url: event.target.value,
      prediction: null
    });
  };

  sendRequest = async () => {
    this.setState({
      is_loading: true
    });
    const query = this.state.youtube_url.split("?")[1];
    const urlParams = queryString.parse(query);
    console.log(urlParams);
    const data = {
      ...this.state
    };
    console.log(data);
    try {
      const res = await axios.post("http://localhost:8000/youtube", data);
      console.log(res.data.prediction);
      this.setState({
        prediction: [...res.data.prediction],
        prediction_original: [...res.data.prediction],
        videoId: urlParams.v,
        is_loading: false
      });
    } catch (error) {
      console.log(error);
    }
  };

  componentDidMount = () => {
    this.readPoliticianList();
  };

  readPoliticianList = () => {
    Papa.parse(csv, {
      download: true,
      header: true,
      complete: this.setPoliticianList
    });
  };
  setPoliticianList = data => {
    const politician_idx2spk = [];
    data.data.map(element => {
      return politician_idx2spk.push(element.name);
    });

    const politician_spk2idx = {};
    for (let i = 0; i <= politician_idx2spk.length; i++) {
      politician_spk2idx[politician_idx2spk[i]] = i;
    }

    this.setState(
      {
        politician_idx2spk: politician_idx2spk,
        politician_spk2idx: politician_spk2idx
      }
    );
  };

  setNewLabel = (event, value) => {
    const prediction = [...this.state.prediction]; // copy prediction
    const idx = Math.floor(this.state.currentTime / 3); // calculate index by current time

    const new_label = this.state.politician_spk2idx[value]; // get label from dictation
    prediction[idx].label = new_label; // put label back in prediction

    this.setState({
      prediction: prediction
    });
  };

  printLabel = () => {
    this.setState(prevState => ({
      newLabel: prevState.prediction
    }));
  };

  saveLabel = () => {
    try {
      const data = {
        "video_id": this.state.videoId,
        "label_list": this.state.prediction
      }
      const res = axios.post("http://localhost:8000/save", data)
      console.log(res.data)
      this.setState({
        is_saved: true
      })
    } catch (error) {
      console.log(error)
    }
  };

  render() {
    let textInput = <CircularProgress />;
    let chart = <Chart />
    if (this.state.prediction && this.state.currentTime) {
      const idx = Math.floor(this.state.currentTime / 3);
      const label = parseInt(this.state.prediction[idx].label);

      textInput = (
        <TextInput
          options={this.state.politician_idx2spk}
          onInputChange={this.setNewLabel}
          defaultValue={this.state.politician_idx2spk[label]}
          key={`movingContactName:${this.state.politician_idx2spk[label]}`}
        />
      );

      let scores = this.state.prediction[idx].scores
      let chart_data = []
      for (const index in scores) {
        const label = scores[index][0]
        const score = scores[index][1]
        
        chart_data.push({
          "name": this.state.politician_idx2spk[parseInt(label)],
          "score": score
        })
      }
      console.log(chart_data)
      chart = (
        <Chart data={chart_data}/>
      )
    }
    let youtube = <p></p>;
    if (this.state.prediction) {
      youtube = (
        <ReactYoutube
          videoId={this.state.videoId}
          setCurrentTimeHandler={this.setCurrentTimeHandler}
        />
      );
    }

    let newLabel = <p></p>;
    if (this.state.newLabel) {
      // console.log(this.state.newLabel)
      newLabel = this.state.newLabel.map(obj => {
        return (
          <div>
            <p>{this.state.politician_idx2spk[parseInt(obj.label)]}</p>
            <p>
              {obj.filename} {obj.label}
            </p>
          </div>
        );
      });
    }

    return (
      <div style={{ padding: 30 }}>
        <Grid container spacing={4} justify="center">
          <Grid
            item
            lg={6}
            sm={12}
            xl={6}
            xs={12}
            alignItems="center"
            justify="center"
          >
            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              {youtube}
            </Box>
            <input
              type="text"
              value={this.state.youtube_url}
              onChange={this.textInputHandler}
              disabled={this.state.is_loading}
            />
            <button
              type="button"
              className="btn btn-primary"
              onClick={this.sendRequest}
            >
              Send!
            </button>
          </Grid>
          <Grid item lg={6} sm={12} xl={6} xs={12} alignItems="center">
            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              <Card text={`Current Time: ${this.state.currentTime}`} />
              <Card text={`Original Predict: ?`} />
              {textInput}
            </Box>
          </Grid>
          <Grid item lg={6} sm={12} xl={6} xs={12} alignItems="center">
            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              <Button
                variant="contained"
                color="primary"
                onClick={this.saveLabel}
              >
                Save!
              </Button>
              {this.state.is_saved ? "Yay" : ":("}
            </Box>
          </Grid>
          <Grid item lg={6} sm={12} xl={6} xs={12} alignItems="center">
            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              {chart}
            </Box>
          </Grid>
        </Grid>
      </div>
    );
  }
}

export default DemoYoutube;
