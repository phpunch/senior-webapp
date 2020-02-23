import React, { Component } from "react";
import "./App.css";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import Home from "./containers/Home";
import Politician from './containers/Politician/Politician'
import Navbar from "./components/Navbar";
import Demo from './containers/Demo'
class App extends Component {
  render() {
    return (
      <Router>
        <Navbar />
        <Switch>
          <Route exact path="/">
            <Demo />
          </Route>
          <Route path="/politician">
            <Politician />
          </Route>
          <Route path="/demo">
            <Home />
          </Route>
        </Switch>
      </Router>
    );
  }
}

export default App;
