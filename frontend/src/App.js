import React from "react";
import "./App.css";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import Home from "./containers/Home";
import Politician from "./containers/Politician/Politician";
import Navbar from "./components/Navbar";
import DemoController from "./containers/Demo/DemoController";
import DemoYoutube from "./containers/DemoYoutube/DemoYoutube";

import CssBaseline from "@material-ui/core/CssBaseline";
import { MuiThemeProvider, createMuiTheme } from "@material-ui/core/styles";

const themeDark = createMuiTheme({
  palette: {
    background: {
      default: "#121212"
    },
    primary: {
      main: "#424242"
    },
    secondary: {
      main: "#03DAC6"
    }
  },
});

const App = () => {
  return (
    <MuiThemeProvider theme={themeDark}>
      <CssBaseline />
      <Router>
        <Navbar />
        <Switch>
          <Route exact path="/">
            <DemoYoutube />
          </Route>
          <Route exact path="/demo">
            <DemoController />
          </Route>
          <Route path="/politician">
            <Politician />
          </Route>
          <Route path="/demo">
            <Home />
          </Route>
        </Switch>
      </Router>
    </MuiThemeProvider>
  );
};

export default App;
