import React, { Component } from "react";
import ReactYoutube from "../../components/ReactYoutube";
import axios from "axios";
import queryString from "query-string"

import Papa from "papaparse";
import csv from "../Politician/politician_list.csv";

class DemoYoutube extends Component {
  state = {
    currentTime: 0,
    youtube_url: "",
    politician_list: null,
    is_loading: false
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
    })
    const query = this.state.youtube_url.split("?")[1]
    const urlParams = queryString.parse(query)
    console.log(urlParams)
    const data = {
      ...this.state
    };
    console.log(data);
    try {
      const res = await axios.post("http://localhost:8000/youtube", data);
      console.log(res.data.prediction);
      this.setState({
        prediction: res.data.prediction,
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
    this.setState(
      {
        politician_list: data.data
      }
    );
  };

  

  render() {
    let name = "";
    if (this.state.prediction && this.state.currentTime) {
      const idx = Math.floor(this.state.currentTime / 3);
      const label = this.state.prediction[idx].label
      name = this.state.politician_list[parseInt(label)].name;
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
    return (
      <div className="container">
        <div className="row">
          <div className="offset-md-3 col-md-6">
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

            {youtube}
            <div className="card card-body">
              Predict: {name}
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default DemoYoutube;
