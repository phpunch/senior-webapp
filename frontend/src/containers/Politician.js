import React, { Component } from "react";
import axios from "axios";
import Papa from "papaparse"
import csv from './df_pass.csv'
class Politician extends Component {
  state = {
    data: null
  };

  readCSV = () => {
    Papa.parse(csv, {
        download: true,
        complete: function(results) {
            return results
        }
    });
  }

  putCSV = () => {
      const results = this.readCSV()
      console.log(results)
      this.setState({
          data: results
      })
  }

  render() {
    let tableData = <p></p>
    if (this.state.data) {
        tableData = this.state.data.map(i => {
            return (<div>
                {i}
            </div>)
        })
    }

    return <div className="container">
        Politician
        <button onClick={this.putCSV}>Click</button>
        {tableData}
        <p>https://docs.google.com/spreadsheets/d/e/2PACX-1vQHs8Rw8KEM6YK0Rce2qWVBVJ96YWTB7tzR0N_T5e6y7C-sLrjMIHl7YPr1aYRDeRA2VnjpIfgIYnUB/pubhtml?gid=1261867508&single=true</p>
        </div>;
  }
}

export default Politician;
