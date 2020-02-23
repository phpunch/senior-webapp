import React from "react";
import YouTube from "react-youtube";

class ReactYoutube extends React.Component {

  state = {
      intervalID: 0
  }
  videoOnReady = (event) => {
    // access to player in all event handlers via event.target
    event.target.pauseVideo();
  }

  videoOnPlay = (event) => {
    const player = event.target
    this.setState({
        intervalID: setInterval(() => {this.props.setCurrentTimeHandler(player.getCurrentTime())}, 1000)
    })
    
  }
  videoOnPause = (event) => {
    clearInterval(this.state.intervalID)
  }
  render() {
    const opts = {
      height: "390",
      width: "640",
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
        onPlay={this.videoOnPlay}
        onPause={this.videoOnPause} 
      />
    );
  }
}

export default ReactYoutube;
