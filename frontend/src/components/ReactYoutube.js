import React from "react";
import YouTube from "react-youtube";

class ReactYoutube extends React.Component {
  state = {
    intervalID: 0
  };
  videoOnReady = event => {
    // access to player in all event handlers via event.target
    event.target.pauseVideo();
  };

  // setCurrentTime = (event) => {
  //   const player = event.target
  //   player.seekTo(this.props.currentTime, true)

  //   // }
  //   // videoOnPause = (event) => {
  //   //   clearInterval(this.state.intervalID)
  //   // }
  // }
  videoOnStateChange = event => {
    clearInterval(this.state.intervalID);
    const player = event.target;

    this.props.getCurrentTimeHandler(player.getCurrentTime())
    this.setState({
      intervalID: setInterval(() => {
        this.props.getCurrentTimeHandler(player.getCurrentTime());
      }, 1000)
    });

  };
  render() {
    const opts = {
      height: "400",
      width: "100%",
      playerVars: {
        // https://developers.google.com/youtube/player_parameters
        autoplay: 1
      }
    };

    return (
      <YouTube
        videoId={this.props.videoId}
        opts={opts}
        onReady={this.videoOnReady}
        // onPlay={this.setCurrentTime}
        // onPause={this.videoOnPause}
        onStateChange={this.videoOnStateChange}
      />
    );
  }
}

export default ReactYoutube;
