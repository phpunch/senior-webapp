import React, { Component } from "react";
import ReactYoutube from "../components/ReactYoutube";
class Demo extends Component {
  state = {
    currentTime: 0
  };

  setCurrentTimeHandler = time => {
    this.setState({
      currentTime: time
    });
  };

  render() {
    return (
      <div className="container">
        <div class="row">
          <div class="offset-md-3 col-md-6">
            <ReactYoutube
              videoId="Y70hiPNxe4Q"
              setCurrentTimeHandler={this.setCurrentTimeHandler}
            />
            <div className="card card-body">
              Current Time: {this.state.currentTime}
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default Demo;
