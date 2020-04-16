import React, { PureComponent } from "react";
import ReactYoutube from "../../components/ReactYoutube";
import axios from "axios";
import queryString from "query-string";

import Papa from "papaparse";
import csv from "../Politician/politician_list.csv";

import { Button, CircularProgress, Grid, Box, TextField, FormControl, Backdrop, Typography } from "@material-ui/core";
import TextInput from "../../components/TextInput";
import Card from "../../components/Card";
import Chart from '../../components/Chart'
import Slider from '../../components/Slider/Slider'
import HorizontalTable from '../../components/HorizontalTable'

function makeid(length) {
  var result = '';
  var characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  var charactersLength = characters.length;
  for (var i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
  }
  return result;
}

const host = '34.71.245.189'

class DemoYoutube extends PureComponent {
  state = {
    currentTime: 0,
    youtube_url:
      "https://www.youtube.com/watch?v=kHk5muJUwuw&feature=emb_title",
    prediction: null,
    politician_idx2spk: [],
    politician_spk2idx: {},
    is_loading: false,
    is_saved: false,
    url_lock: false,
    error_msg: null,
  };

  getCurrentTimeHandler = time => {
    this.setState({
      currentTime: time
    });
  };

  // setCurrentTimeHandler = time => {
  //   this.setState({
  //     currentTime: time
  //   })
  // }

  textInputHandler = event => {
    this.setState({
      youtube_url: event.target.value,
      prediction: null
    });
  };

  sendRequest = async () => {
    this.setState({
      is_loading: true,
      url_lock: true
    });
    const query = this.state.youtube_url.split("?")[1];
    const urlParams = queryString.parse(query);
    console.log(urlParams);
    const data = {
      youtube_url: this.state.youtube_url,
      ticket: makeid(5)
    };
    console.log(data);
    try {
      const res = await axios.post(`http://${host}:8000/youtube`, data);
      console.log(res.data.prediction);
      this.setState({
        prediction: [...res.data.prediction],
        videoId: urlParams.v,
        is_loading: false,
      });
    } catch (error) {
      console.log(error);
      this.setState({
        error_msg: error.message
      })
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

    if (value === "") {
      prediction[idx].label = ""
    } else {
      prediction[idx].label = this.state.politician_spk2idx[value]; // change the label
    }
    // console.log(prediction[idx].label)

    this.setState({
      prediction: prediction
    });
  };

  printLabel = () => {
    this.setState(prevState => ({
      newLabel: prevState.prediction
    }));
  };

  saveLabel = async () => {
    try {
      this.setState({
        is_loading: true
      })
      const data = {
        ticket: makeid(5),
        youtube_url: this.state.youtube_url,
        video_id: this.state.videoId,
        label_list: this.state.prediction
      }
      const res = await axios.post(`http://${host}:8000/save`, data)
      console.log(res.data)
      this.setState({
        is_loading: false,
        is_saved: true
      })
    } catch (error) {
      console.log(error)
      this.setState({
        error_msg: error.message
      })
    }
  };

  render() {
    let old_label = ""
    let textInput = <p></p>
    let chart = <Chart />
    let slider = <p></p>
    let youtube = <p></p>;
    let horizontal_table = "";

    if (this.state.prediction && this.state.currentTime) {
      const idx = Math.floor(this.state.currentTime / 3);
      const label_id = parseInt(this.state.prediction[idx].label);

      textInput = (
        <TextInput
          options={this.state.politician_idx2spk}
          onInputChange={this.setNewLabel}
          defaultValue={this.state.politician_idx2spk[label_id]}
          key={`movingContactName:${this.state.politician_idx2spk[label_id]}`}
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

      old_label = this.state.politician_idx2spk[parseInt(scores[0][0])]

      // console.log(chart_data)
      chart = (
        <Chart data={chart_data} />
      )

      const slider_list = this.state.prediction.map((frame, index) => {
        return {
          'filename': frame.filename,
          'time': "วินาที : " + (index * 3).toString(), 
          'name': this.state.politician_idx2spk[parseInt(frame.label)]
        }
      })
      // console.log(slider_list)
      slider = <Slider
        list={slider_list}
        selected={this.state.prediction[idx].filename}
      />
      
      const min_bound = Math.max(0, idx - 2)
      const max_bound = Math.min(this.state.prediction.length - 1, idx + 2)
      // console.log(min_bound, max_bound)
      let horizontal_table_list = [];
      let count = 0
      if (parseInt(idx) < 1) {
        horizontal_table_list.unshift({})
      }
      if (parseInt(idx) < 2) {
        horizontal_table_list.unshift({})
      }
      // let index = 0
      // while (index < slider_list.length) {
      //   console.log(min_bound, index, max_bound)
      //   if (count < 2 && check < 2) {
      //     console.log("YES")
      //     horizontal_table_list.push({})
      //     count++;
      //   }
      //   else if (min_bound <= index && index <= max_bound) {
          
      //     horizontal_table_list.push(slider_list[index])
      //     count++;
      //     index++;
      //   }
      //   if (count === 5) {
      //     break
      //   }

      // }


      for (const index in slider_list) {
        
        if (min_bound <= parseInt(index) && parseInt(index) <= max_bound) {
          console.log(min_bound, parseInt(index), max_bound)
          horizontal_table_list.push(slider_list[index])
        }
      }
      console.log(horizontal_table_list)
      horizontal_table = <HorizontalTable rows={horizontal_table_list}/>

    }
    if (this.state.prediction) {
      youtube = (
        <ReactYoutube
          videoId={this.state.videoId}
          getCurrentTimeHandler={this.getCurrentTimeHandler}
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
            <p>-------------------------------</p>
          </div>
        );
      });
    }

    let loading_message = <CircularProgress color="primary" />
    if (this.state.error_msg) {
      loading_message = <p>{this.state.error_msg}</p>
    }

    return (
      <div style={{ padding: 30 }}>
        <Grid container spacing={4} justify="center">
          <Grid
            item
            lg={12}
            sm={12}
            xl={12}
            xs={12}>

            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              <FormControl fullWidth>
                <TextField
                  id="youtube_url"
                  label="Youtube URL"
                  value={this.state.youtube_url}
                  onChange={this.textInputHandler}
                  disabled={this.state.url_lock} />
              </FormControl>

              <Button
                variant="contained"
                color="secondary"
                onClick={this.sendRequest}
                disabled={this.state.url_lock}
              >
                Send!
              </Button>
            </Box>
          </Grid>
          <Grid
            item
            lg={6}
            sm={12}
            xl={6}
            xs={12}
          >
            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              {youtube}
            </Box>

          </Grid>
          <Grid item lg={6} sm={12} xl={6} xs={12}>
            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              <Card text={`Current Time: ${this.state.currentTime}`} />
              <Card text={`Original Predict: ${old_label}`} />
              {textInput}
            </Box>
            <Box borderRadius={16} bgcolor="background.paper" p={5}>

            </Box>
          </Grid>
          <Grid item lg={12} sm={12} xl={12} xs={12}>
            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              {/* {slider} */}
              {horizontal_table}
            </Box>
          </Grid>
          <Grid item lg={6} sm={12} xl={6} xs={12}>
            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              <Button
                variant="contained"
                color="primary"
                onClick={this.saveLabel}
              >
                Save!
              </Button>
              {this.state.is_saved ? "Saved!" : ":("}
              {/* <Button
                variant="contained"
                color="primary"
                onClick={this.printLabel}
              >
                Print!
              </Button> */}
              {newLabel}
            </Box>
          </Grid>
          <Grid item lg={6} sm={12} xl={6} xs={12} >
            <Box borderRadius={16} bgcolor="background.paper" p={5}>
              {chart}
            </Box>
          </Grid>
        </Grid>
        <Backdrop style={{ zIndex: 999, color: '#000000', }} open={this.state.is_loading}>
          <Box
            borderRadius={16}
            bgcolor="background.paper"
            alignItems="center"
            justifyContent="center"
            p={5}>
            <Typography variant="h5" >
              เราจะทำตามสัญญา ขอเวลาอีกไม่นาน ...
            </Typography>
            {loading_message}
          </Box>
        </Backdrop>
      </div >
    );
  }
}

export default DemoYoutube;
