import React, { Component } from "react";
import Papa from "papaparse";
import csv from "./politician_list.csv";
import { Button, CircularProgress, Grid, Box } from "@material-ui/core";

class Politician extends Component {
  state = {
    data: null
  };

  componentDidMount = () => {
    this.readCSV();
  };

  setData = data => {
    this.setState({
      data: data.data
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
    let table = (
      <div>
        <p>Loading ... </p>
      </div>
    );
    if (this.state.data) {
      const tableData = this.state.data.map(person => {
        return (
          <tr>
            <th scope="row">{person["index"]}</th>
            <td>{person["name"]}</td>
          </tr>
        );
      });
      table = (
        <table className="table">
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Name</th>
            </tr>
          </thead>
          <tbody>{tableData}</tbody>
        </table>
      );
    }

    return (
      <div style={{ padding: 20 }}>
        <Grid container justify="center">
          <Box borderRadius={16} bgcolor="background.paper" p={5}>
            politician_list
            {table}
          </Box>
        </Grid>
      </div>
    );
  }
}

export default Politician;
