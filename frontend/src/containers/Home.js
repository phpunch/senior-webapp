import React, { Component } from "react";
import axios from "axios";
import ReactYoutube from "../components/ReactYoutube";
class Home extends Component {
  state = {
    selectedFile: null,
    isProcess: false,
    predicted_output: null
  };
  onChangeHandler = event => {
    this.setState({
      selectedFile: event.target.files[0],
      loaded: 0
    });
  };

  onClickHandler = async () => {
    const data = new FormData();
    data.append("file", this.state.selectedFile);
    this.setState({
      isProcess: true
    })
    try {
      const res = await axios.post("http://localhost:8000/upload", data);
      this.setState({
        predicted_output: res.data.predicted_output
      })
      console.log(res.data);
    } catch(e) {
      console.log(e)
    }
    this.setState({
      isProcess: false
    })
  };

  render() {
    let loader = <p></p>
    if (this.state.isProcess) {
      loader = <p>Process your request</p>
    }
    return (
      <div className="container">
        <div class="row">
          <div class="offset-md-3 col-md-6">
            <div class="form-group files">
              <label>Upload Your File</label>
              <input type="file" name="file" onChange={this.onChangeHandler} />
            </div>
            <button
              type="button"
              class="btn btn-primary btn-lg"
              onClick={this.onClickHandler}
              disabled={this.state.isProcess}
            >
              Upload
            </button>
            {loader}
            <p>Predict : {this.state.predicted_output}</p>

            <div>
              <ReactYoutube videoId="Y70hiPNxe4Q"/>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default Home;
