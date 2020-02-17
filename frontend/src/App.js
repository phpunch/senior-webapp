import React, {Component} from 'react';
import './App.css';
import axios from 'axios'

class App extends Component {
  state = {
    selectedFile: null
  }
  onChangeHandler = (event) => {
    this.setState({
      selectedFile: event.target.files[0],
      loaded: 0,
    })
  }

  onClickHandler = async () => {
    const data = new FormData()
    data.append('file', this.state.selectedFile)
    const res = await axios.post("http://localhost:8000/upload", data)
    console.log(res.data)
  }

  render(){
    return (
      <div className="container">
        <div class="row">
          <div class="offset-md-3 col-md-6">
            <div class="form-group files">
              <label>Upload Your File</label>
              <input type="file" name="file" onChange={this.onChangeHandler} />
            </div>
            <button type="button" class="btn btn-small" onClick={this.onClickHandler}>Upload</button>
          </div>
        </div>
      </div>
    );
  }
  
}

export default App;
