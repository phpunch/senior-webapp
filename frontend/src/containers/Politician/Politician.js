import React, { Component } from "react";
import Papa from "papaparse";
import csv from "./politician_list.csv";
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
      <div className="container">
        politician_list
        {table}
      </div>
    );
  }
}

export default Politician;
