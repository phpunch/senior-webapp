import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableContainer from "@material-ui/core/TableContainer";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import Paper from "@material-ui/core/Paper";

const useStyles = makeStyles({
  table: {
    minWidth: 650,
  },
});

// const rows = [
//   { filename: "demo-0001", label: "192" },
//   { filename: "demo-0002", label: "112" },
//   { filename: "demo-0003", label: "122" },
//   { filename: "demo-0004", label: "234" },
//   { filename: "demo-0005", label: "234" },
// ];

export default function HorizontalTable(props) {
  const classes = useStyles();
  const rows = props.rows
  return (
    <TableContainer component={Paper}>
      <Table className={classes.table} size="small" aria-label="a dense table">
        <TableHead>
          <TableRow>
            <TableCell style={{minWidth:100, maxWidth:100}}>{rows[0].time ? rows[0].time : ""}</TableCell>
            <TableCell style={{minWidth:100, maxWidth:100}}>{rows[1].time ? rows[1].time : ""}</TableCell>
            <TableCell style={{minWidth:100, maxWidth:100, color: "white", backgroundColor: "black"}}>{rows[2].time}</TableCell>
            <TableCell style={{minWidth:100, maxWidth:100}}>{rows[3] ? rows[3].time : ""}</TableCell>
            <TableCell style={{minWidth:100, maxWidth:100}}>{rows[4] ? rows[4].time : ""}</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell style={{minWidth:100, maxWidth:100}}>{rows[0].name ? rows[0].name: ""}</TableCell>
            <TableCell style={{minWidth:100, maxWidth:100}}>{rows[1].name ? rows[1].name: ""}</TableCell>
            <TableCell style={{minWidth:100, maxWidth:100}}>{rows[2].name}</TableCell>
            <TableCell style={{minWidth:100, maxWidth:100}}>{rows[3] ? rows[3].name: ""}</TableCell>
            <TableCell style={{minWidth:100, maxWidth:100}}>{rows[4] ? rows[4].name: ""}</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </TableContainer>
  );
}
