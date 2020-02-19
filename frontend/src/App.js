import React, { Component } from "react";
import "./App.css";
import { BrowserRouter as Router, Switch, Route, Link } from "react-router-dom";
import Home from "./containers/Home";
import Politician from './containers/Politician'
import Navbar from "./components/Navbar";
class App extends Component {
  render() {
    return (
      <Router>
        <Navbar />
        <Switch>
          <Route exact path="/">
            <Home />
          </Route>
          <Route path="/politician">
            <Politician />
          </Route>
        </Switch>
      </Router>
    );
  }
}

export default App;
