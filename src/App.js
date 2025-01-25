
// import React, { useState } from 'react';
// import axios from 'axios';

// function App() {
//   const [sourceFile, setSourceFile] = useState(null);
//   const [targetFile, setTargetFile] = useState(null);
//   const [metrics, setMetrics] = useState(null);
//   const [targetData, setTargetData] = useState([]);

//   const handleSourceFileChange = (e) => {
//     setSourceFile(e.target.files[0]);
//   };

//   const handleTargetFileChange = (e) => {
//     setTargetFile(e.target.files[0]);
//   };

//   const handleSubmit = async (modelType) => {
//     const formData = new FormData();
//     formData.append('source', sourceFile);
//     formData.append('target', targetFile);
//     formData.append('model_type', modelType);

//     try {
//       const response = await axios.post('http://localhost:5000/upload', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });
//       setMetrics(response.data);
//       setTargetData(response.data.targetDataWithLabels);
//     } catch (error) {
//       console.error('Error uploading files:', error);
//     }
//   };

//   return (
//     <div className="App">
//       <h1>Upload Source and Target Files</h1>
//       <input type="file" onChange={handleSourceFileChange} />
//       <input type="file" onChange={handleTargetFileChange} />
//       <button onClick={() => handleSubmit('pls')}>Submit PLS</button>
//       <button onClick={() => handleSubmit('dpls')}>Submit DPLS</button>
//       <button onClick={() => handleSubmit('gdpls')}>Submit GDPLS</button>
//       <button onClick={() => handleSubmit('coral')}>Submit CORAL</button>
//       {metrics && (
//         <div>
//           <h2>General Metrics:</h2>
//           <p>Accuracy: {metrics.accuracy}</p>
//           <p>AUC-ROC: {metrics.aucRoc}</p>
//           <p>F1-Score: {metrics.f1Score}</p>
//         </div>
//       )}
//       {targetData.length > 0 && (
//         <div>
//           <h2>Target Data with Labels</h2>
//           <table>
//             <thead>
//               <tr>
//                 {Object.keys(targetData[0]).map((key) => (
//                   <th key={key}>{key}</th>
//                 ))}
//               </tr>
//             </thead>
//             <tbody>
//               {targetData.map((row, index) => (
//                 <tr key={index}>
//                   {Object.values(row).map((value, i) => (
//                     <td key={i}>{value}</td>
//                   ))}
//                 </tr>
//               ))}
//             </tbody>
//           </table>
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;



/// STYLE    1

// import React, { useState } from 'react';
// import axios from 'axios';
// import { Container, Typography, Button, Input, Card, CardContent, Table, TableBody, TableCell, TableHead, TableRow, Box } from '@mui/material';

// function App() {
//   const [sourceFile, setSourceFile] = useState(null);
//   const [targetFile, setTargetFile] = useState(null);
//   const [metrics, setMetrics] = useState(null);
//   const [targetData, setTargetData] = useState([]);

//   const handleSourceFileChange = (e) => {
//     setSourceFile(e.target.files[0]);
//   };

//   const handleTargetFileChange = (e) => {
//     setTargetFile(e.target.files[0]);
//   };

//   const handleSubmit = async (modelType) => {
//     const formData = new FormData();
//     formData.append('source', sourceFile);
//     formData.append('target', targetFile);
//     formData.append('model_type', modelType);

//     try {
//       const response = await axios.post('http://localhost:5000/upload', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });
//       setMetrics(response.data);
//       setTargetData(response.data.targetDataWithLabels);
//     } catch (error) {
//       console.error('Error uploading files:', error);
//     }
//   };

//   return (
//     <Container maxWidth="md" style={{ marginTop: '50px' }}>
//       <Card variant="outlined" style={{ padding: '20px' }}>
//         <CardContent>
//           <Typography variant="h4" gutterBottom>
//             Upload Source and Target Files
//           </Typography>

//           <Box mb={2}>
//             <Input type="file" onChange={handleSourceFileChange} fullWidth={true} />
//           </Box>
//           <Box mb={2}>
//             <Input type="file" onChange={handleTargetFileChange} fullWidth={true} />
//           </Box>

//           <Box display="flex" justifyContent="space-between" mt={2} mb={2}>
//             <Button variant="contained" color="primary" onClick={() => handleSubmit('pls')}>
//               Submit PLS
//             </Button>
//             <Button variant="contained" color="secondary" onClick={() => handleSubmit('dpls')}>
//               Submit DPLS
//             </Button>
//             <Button variant="contained" color="success" onClick={() => handleSubmit('gdpls')}>
//               Submit GDPLS
//             </Button>
//             <Button variant="contained" color="error" onClick={() => handleSubmit('coral')}>
//               Submit CORAL
//             </Button>
//           </Box>

//           {metrics && (
//             <Box mt={4}>
//               <Typography variant="h5" gutterBottom>
//                 General Metrics:
//               </Typography>
//               <Typography>Accuracy: {metrics.accuracy}</Typography>
//               <Typography>AUC-ROC: {metrics.aucRoc}</Typography>
//               <Typography>F1-Score: {metrics.f1Score}</Typography>
//             </Box>
//           )}

//           {targetData.length > 0 && (
//             <Box mt={4}>
//               <Typography variant="h5" gutterBottom>
//                 Target Data with Labels
//               </Typography>
//               <Table>
//                 <TableHead>
//                   <TableRow>
//                     {Object.keys(targetData[0]).map((key) => (
//                       <TableCell key={key}>{key}</TableCell>
//                     ))}
//                   </TableRow>
//                 </TableHead>
//                 <TableBody>
//                   {targetData.map((row, index) => (
//                     <TableRow key={index}>
//                       {Object.values(row).map((value, i) => (
//                         <TableCell key={i}>{value}</TableCell>
//                       ))}
//                     </TableRow>
//                   ))}
//                 </TableBody>
//               </Table>
//             </Box>
//           )}
//         </CardContent>
//       </Card>
//     </Container>
//   );
// }

// export default App;


//  main main 
// import React, { useState } from 'react';
// import axios from 'axios';
// import { Container, Typography, Button, Input, Card, CardContent, Table, TableBody, TableCell, TableHead, TableRow, Box, Grid, Paper, Select, MenuItem, FormControl, InputLabel, CircularProgress } from '@mui/material';
// import CheckCircleIcon from '@mui/icons-material/CheckCircle';
// import AssessmentIcon from '@mui/icons-material/Assessment';
// import BarChartIcon from '@mui/icons-material/BarChart';

// function App() {
//   const [sourceFile, setSourceFile] = useState(null);
//   const [targetFile, setTargetFile] = useState(null);
//   const [metrics, setMetrics] = useState(null);
//   const [targetData, setTargetData] = useState([]);
//   const [selectedModel, setSelectedModel] = useState('pls');
//   const [loading, setLoading] = useState(false);

//   const handleSourceFileChange = (e) => {
//     setSourceFile(e.target.files[0]);
//   };

//   const handleTargetFileChange = (e) => {
//     setTargetFile(e.target.files[0]);
//   };

//   const handleModelChange = (e) => {
//     setSelectedModel(e.target.value);
//   };

//   const handleSubmit = async () => {
//     setLoading(true);
//     const formData = new FormData();
//     formData.append('source', sourceFile);
//     formData.append('target', targetFile);
//     formData.append('model_type', selectedModel);

//     try {
//       const response = await axios.post('http://localhost:5000/upload', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });
//       setMetrics(response.data);
//       setTargetData(response.data.targetDataWithLabels);
//     } catch (error) {
//       console.error('Error uploading files:', error);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <Container maxWidth="md" style={{ marginTop: '50px' }}>
//       <Card variant="outlined" style={{ padding: '20px' }}>
//         <CardContent>
//           <Typography variant="h4" gutterBottom>
//             Upload Source and Target Files
//           </Typography>

//           <Box mb={2}>
//             <Input type="file" onChange={handleSourceFileChange} fullWidth={true} />
//           </Box>
//           <Box mb={2}>
//             <Input type="file" onChange={handleTargetFileChange} fullWidth={true} />
//           </Box>

//           <Box mb={2}>
//             <FormControl fullWidth>
//               <InputLabel>Choose Method</InputLabel>
//               <Select
//                 value={selectedModel}
//                 onChange={handleModelChange}
//                 label="Choose Method"
//               >
//                 <MenuItem value="pls">PLS</MenuItem>
//                 <MenuItem value="dpls">DPLS</MenuItem>
//                 <MenuItem value="gdpls">GDPLS</MenuItem>
//                 <MenuItem value="coral">CORAL</MenuItem>
//                 <MenuItem value="tca">TCA</MenuItem>
//                 <MenuItem value="tcaPlus">TCA_PLUS</MenuItem>
//                 <MenuItem value="jda">JDA</MenuItem>
//               </Select>
//             </FormControl>
//           </Box>

//           <Box mt={2} mb={2}>
//             <Button variant="contained" color="primary" onClick={handleSubmit}>
//               Submit
//             </Button>
//           </Box>

//           {loading && (
//             <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
//               <CircularProgress />
//             </Box>
//           )}

//           {!loading && metrics && (
//             <Box mt={4}>
//               <Typography variant="h5" gutterBottom>
//                 General Metrics:
//               </Typography>
//               <Grid container spacing={3}>
//                 <Grid item xs={4}>
//                   <Paper elevation={3} style={{ padding: '20px', textAlign: 'center' }}>
//                     <CheckCircleIcon fontSize="large" color="success" />
//                     <Typography variant="h6" gutterBottom>
//                       Accuracy
//                     </Typography>
//                     <Typography variant="body1">{metrics.accuracy}</Typography>
//                   </Paper>
//                 </Grid>
//                 <Grid item xs={4}>
//                   <Paper elevation={3} style={{ padding: '20px', textAlign: 'center' }}>
//                     <AssessmentIcon fontSize="large" color="primary" />
//                     <Typography variant="h6" gutterBottom>
//                       AUC-ROC
//                     </Typography>
//                     <Typography variant="body1">{metrics.aucRoc}</Typography>
//                   </Paper>
//                 </Grid>
//                 <Grid item xs={4}>
//                   <Paper elevation={3} style={{ padding: '20px', textAlign: 'center' }}>
//                     <BarChartIcon fontSize="large" color="secondary" />
//                     <Typography variant="h6" gutterBottom>
//                       F1-Score
//                     </Typography>
//                     <Typography variant="body1">{metrics.f1Score}</Typography>
//                   </Paper>
//                 </Grid>
//               </Grid>
//             </Box>
//           )}

//           {!loading && targetData.length > 0 && (
//             <Box mt={4}>
//               <Typography variant="h5" gutterBottom>
//                 Target Data with Labels
//               </Typography>
//               <Table>
//                 <TableHead>
//                   <TableRow>
//                     {Object.keys(targetData[0]).map((key) => (
//                       <TableCell key={key}>{key}</TableCell>
//                     ))}
//                   </TableRow>
//                 </TableHead>
//                 <TableBody>
//                   {targetData.map((row, index) => (
//                     <TableRow key={index}>
//                       {Object.values(row).map((value, i) => (
//                         <TableCell key={i}>{value}</TableCell>
//                       ))}
//                     </TableRow>
//                   ))}
//                 </TableBody>
//               </Table>
//             </Box>
//           )}
//         </CardContent>
//       </Card>
//     </Container>
//   );
// }

// export default App;










// STYLE 2   WITH ICON
// import React, { useState } from 'react';
// import axios from 'axios';
// import { Container, Typography, Button, Input, Card, CardContent, Table, TableBody, TableCell, TableHead, TableRow, Box, Grid, Paper } from '@mui/material';
// import CheckCircleIcon from '@mui/icons-material/CheckCircle';
// import AssessmentIcon from '@mui/icons-material/Assessment';
// import BarChartIcon from '@mui/icons-material/BarChart';

// function App() {
//   const [sourceFile, setSourceFile] = useState(null);
//   const [targetFile, setTargetFile] = useState(null);
//   const [metrics, setMetrics] = useState(null);
//   const [targetData, setTargetData] = useState([]);

//   const handleSourceFileChange = (e) => {
//     setSourceFile(e.target.files[0]);
//   };

//   const handleTargetFileChange = (e) => {
//     setTargetFile(e.target.files[0]);
//   };

//   const handleSubmit = async (modelType) => {
//     const formData = new FormData();
//     formData.append('source', sourceFile);
//     formData.append('target', targetFile);
//     formData.append('model_type', modelType);

//     try {
//       const response = await axios.post('http://localhost:5000/upload', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });
//       setMetrics(response.data);
//       setTargetData(response.data.targetDataWithLabels);
//     } catch (error) {
//       console.error('Error uploading files:', error);
//     }
//   };

//   return (
//     <Container maxWidth="md" style={{ marginTop: '50px' }}>
//       <Card variant="outlined" style={{ padding: '20px' }}>
//         <CardContent>
//           <Typography variant="h4" gutterBottom>
//             Upload Source and Target Files
//           </Typography>

//           <Box mb={2}>
//             <Input type="file" onChange={handleSourceFileChange} fullWidth={true} />
//           </Box>
//           <Box mb={2}>
//             <Input type="file" onChange={handleTargetFileChange} fullWidth={true} />
//           </Box>

//           <Box display="flex" justifyContent="space-between" mt={2} mb={2}>
//             <Button variant="contained" color="primary" onClick={() => handleSubmit('pls')}>
//               Submit PLS
//             </Button>
//             <Button variant="contained" color="secondary" onClick={() => handleSubmit('dpls')}>
//               Submit DPLS
//             </Button>
//             <Button variant="contained" color="success" onClick={() => handleSubmit('gdpls')}>
//               Submit GDPLS
//             </Button>
//             <Button variant="contained" color="error" onClick={() => handleSubmit('coral')}>
//               Submit CORAL
//             </Button>
//           </Box>

//           {metrics && (
//             <Box mt={4}>
//               <Typography variant="h5" gutterBottom>
//                 General Metrics:
//               </Typography>
//               <Grid container spacing={3}>
//                 <Grid item xs={4}>
//                   <Paper elevation={3} style={{ padding: '20px', textAlign: 'center' }}>
//                     <CheckCircleIcon fontSize="large" color="success" />
//                     <Typography variant="h6" gutterBottom>
//                       Accuracy
//                     </Typography>
//                     <Typography variant="body1">{metrics.accuracy}</Typography>
//                   </Paper>
//                 </Grid>
//                 <Grid item xs={4}>
//                   <Paper elevation={3} style={{ padding: '20px', textAlign: 'center' }}>
//                     <AssessmentIcon fontSize="large" color="primary" />
//                     <Typography variant="h6" gutterBottom>
//                       AUC-ROC
//                     </Typography>
//                     <Typography variant="body1">{metrics.aucRoc}</Typography>
//                   </Paper>
//                 </Grid>
//                 <Grid item xs={4}>
//                   <Paper elevation={3} style={{ padding: '20px', textAlign: 'center' }}>
//                     <BarChartIcon fontSize="large" color="secondary" />
//                     <Typography variant="h6" gutterBottom>
//                       F1-Score
//                     </Typography>
//                     <Typography variant="body1">{metrics.f1Score}</Typography>
//                   </Paper>
//                 </Grid>
//               </Grid>
//             </Box>
//           )}

//           {targetData.length > 0 && (
//             <Box mt={4}>
//               <Typography variant="h5" gutterBottom>
//                 Target Data with Labels
//               </Typography>
//               <Table>
//                 <TableHead>
//                   <TableRow>
//                     {Object.keys(targetData[0]).map((key) => (
//                       <TableCell key={key}>{key}</TableCell>
//                     ))}
//                   </TableRow>
//                 </TableHead>
//                 <TableBody>
//                   {targetData.map((row, index) => (
//                     <TableRow key={index}>
//                       {Object.values(row).map((value, i) => (
//                         <TableCell key={i}>{value}</TableCell>
//                       ))}
//                     </TableRow>
//                   ))}
//                 </TableBody>
//               </Table>
//             </Box>
//           )}
//         </CardContent>
//       </Card>
//     </Container>
//   );
// }

// export default App;








// STYLE 3   SCROLLING
// import React, { useState } from 'react';
// import axios from 'axios';
// import { Container, Typography, Button, Input, Card, CardContent, Table, TableBody, TableCell, TableHead, TableRow, Box, Pagination } from '@mui/material';

// function App() {
//   const [sourceFile, setSourceFile] = useState(null);
//   const [targetFile, setTargetFile] = useState(null);
//   const [metrics, setMetrics] = useState(null);
//   const [targetData, setTargetData] = useState([]);
//   const [currentPage, setCurrentPage] = useState(1);
//   const itemsPerPage = 10; // Number of rows per page

//   const handleSourceFileChange = (e) => {
//     setSourceFile(e.target.files[0]);
//   };

//   const handleTargetFileChange = (e) => {
//     setTargetFile(e.target.files[0]);
//   };

//   const handleSubmit = async (modelType) => {
//     const formData = new FormData();
//     formData.append('source', sourceFile);
//     formData.append('target', targetFile);
//     formData.append('model_type', modelType);

//     try {
//       const response = await axios.post('http://localhost:5000/upload', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });
//       setMetrics(response.data);
//       setTargetData(response.data.targetDataWithLabels);
//     } catch (error) {
//       console.error('Error uploading files:', error);
//     }
//   };

//   const handlePageChange = (event, value) => {
//     setCurrentPage(value);
//   };

//   // Paginate targetData
//   const indexOfLastItem = currentPage * itemsPerPage;
//   const indexOfFirstItem = indexOfLastItem - itemsPerPage;
//   const currentData = targetData.slice(indexOfFirstItem, indexOfLastItem);

//   return (
//     <Container maxWidth="md" style={{ marginTop: '50px' }}>
//       <Card variant="outlined" style={{ padding: '20px' }}>
//         <CardContent>
//           <Typography variant="h4" gutterBottom>
//             Upload Source and Target Files
//           </Typography>

//           <Box mb={2}>
//             <Input type="file" onChange={handleSourceFileChange} fullWidth={true} />
//           </Box>
//           <Box mb={2}>
//             <Input type="file" onChange={handleTargetFileChange} fullWidth={true} />
//           </Box>

//           <Box display="flex" justifyContent="space-between" mt={2} mb={2}>
//             <Button variant="contained" color="primary" onClick={() => handleSubmit('pls')}>
//               Submit PLS
//             </Button>
//             <Button variant="contained" color="secondary" onClick={() => handleSubmit('dpls')}>
//               Submit DPLS
//             </Button>
//             <Button variant="contained" color="success" onClick={() => handleSubmit('gdpls')}>
//               Submit GDPLS
//             </Button>
//             <Button variant="contained" color="error" onClick={() => handleSubmit('coral')}>
//               Submit CORAL
//             </Button>
//           </Box>

//           {metrics && (
//             <Box mt={4}>
//               <Typography variant="h5" gutterBottom>
//                 General Metrics:
//               </Typography>
//               <Typography>Accuracy: {metrics.accuracy}</Typography>
//               <Typography>AUC-ROC: {metrics.aucRoc}</Typography>
//               <Typography>F1-Score: {metrics.f1Score}</Typography>
//             </Box>
//           )}

//           {targetData.length > 0 && (
//             <Box mt={4}>
//               <Typography variant="h5" gutterBottom>
//                 Target Data with Labels
//               </Typography>
//               <Table>
//                 <TableHead>
//                   <TableRow>
//                     {Object.keys(targetData[0]).map((key) => (
//                       <TableCell key={key}>{key}</TableCell>
//                     ))}
//                   </TableRow>
//                 </TableHead>
//                 <TableBody>
//                   {currentData.map((row, index) => (
//                     <TableRow key={index}>
//                       {Object.values(row).map((value, i) => (
//                         <TableCell key={i}>{value}</TableCell>
//                       ))}
//                     </TableRow>
//                   ))}
//                 </TableBody>
//               </Table>
//               <Box mt={2} display="flex" justifyContent="center">
//                 <Pagination
//                   count={Math.ceil(targetData.length / itemsPerPage)}
//                   page={currentPage}
//                   onChange={handlePageChange}
//                   color="primary"
//                 />
//               </Box>
//             </Box>
//           )}
//         </CardContent>
//       </Card>
//     </Container>
//   );
// }

// export default App;




// #   experiment    sada
// import React, { useState } from 'react';
// import axios from 'axios';
// import { Container, Typography, Button, Input, Card, CardContent, Table, TableBody, TableCell, TableHead, TableRow, Box, Grid, Paper, Select, MenuItem, FormControl, InputLabel, CircularProgress, Grow } from '@mui/material';
// import CheckCircleIcon from '@mui/icons-material/CheckCircle';
// import AssessmentIcon from '@mui/icons-material/Assessment';
// import BarChartIcon from '@mui/icons-material/BarChart';
// import FileUploadIcon from '@mui/icons-material/FileUpload';
// import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
// import Fade from '@mui/material/Fade';
// import { createTheme, ThemeProvider } from '@mui/material/styles';

// const theme = createTheme({
//   palette: {
//     primary: {
//       main: '#673ab7',
//     },
//     secondary: {
//       main: '#ff9800',
//     },
//     success: {
//       main: '#4caf50',
//     },
//     error: {
//       main: '#f44336',
//     },
//   },
//   typography: {
//     fontFamily: 'Roboto, Arial, sans-serif',
//   },
// });

// function App() {
//   const [sourceFile, setSourceFile] = useState(null);
//   const [targetFile, setTargetFile] = useState(null);
//   const [metrics, setMetrics] = useState(null);
//   const [targetData, setTargetData] = useState([]);
//   const [selectedModel, setSelectedModel] = useState('pls');
//   const [loading, setLoading] = useState(false);

//   const handleSourceFileChange = (e) => {
//     setSourceFile(e.target.files[0]);
//   };

//   const handleTargetFileChange = (e) => {
//     setTargetFile(e.target.files[0]);
//   };

//   const handleModelChange = (e) => {
//     setSelectedModel(e.target.value);
//   };

//   const handleSubmit = async () => {
//     setLoading(true);
//     const formData = new FormData();
//     formData.append('source', sourceFile);
//     formData.append('target', targetFile);
//     formData.append('model_type', selectedModel);

//     try {
//       const response = await axios.post('http://localhost:5000/upload', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });
//       setMetrics(response.data);
//       setTargetData(response.data.targetDataWithLabels);
//     } catch (error) {
//       console.error('Error uploading files:', error);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <ThemeProvider theme={theme}>
//       <Container maxWidth="md" style={{ marginTop: '50px' }}>
//         {/* Title Section */}
//         <Grow in={true} timeout={1000}>
//           <Card variant="outlined" style={{ padding: '30px', background: '#f5f5f5', boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)' }}>
//             <CardContent style={{ textAlign: 'center' }}>
//               <Typography variant="h3" color="primary" style={{ fontWeight: 600, marginBottom: '20px' }}>
//                 Transfer Learning Tool
//               </Typography>
//               <Typography variant="subtitle1" color="textSecondary">
//                 Easily upload your source and target datasets, choose your preferred model, and let the tool do the magic!
//               </Typography>
//             </CardContent>
//           </Card>
//         </Grow>

//         {/* Upload Section */}
//         <Fade in={true} timeout={1000}>
//           <Card variant="outlined" style={{ marginTop: '40px', padding: '20px', background: '#fff', boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)' }}>
//             <CardContent>
//               <Typography variant="h4" color="secondary" gutterBottom>
//                 Upload Source and Target Files
//               </Typography>
//               <Box display="flex" justifyContent="center" alignItems="center" mb={2}>
//                 <FileUploadIcon fontSize="large" color="primary" />
//               </Box>

//               <Box mb={2}>
//                 <Input type="file" onChange={handleSourceFileChange} fullWidth={true} inputProps={{ style: { padding: '10px' } }} />
//               </Box>
//               <Box mb={2}>
//                 <Input type="file" onChange={handleTargetFileChange} fullWidth={true} inputProps={{ style: { padding: '10px' } }} />
//               </Box>

//               <Box mb={2}>
//                 <FormControl fullWidth>
//                   <InputLabel>Choose Method</InputLabel>
//                   <Select
//                     value={selectedModel}
//                     onChange={handleModelChange}
//                     label="Choose Method"
//                     style={{ padding: '10px' }}
//                   >
//                     <MenuItem value="pls">PLS</MenuItem>
//                     <MenuItem value="dpls">DPLS</MenuItem>
//                     <MenuItem value="gdpls">GDPLS</MenuItem>
//                     <MenuItem value="coral">CORAL</MenuItem>
//                     <MenuItem value="tca">TCA</MenuItem>
//                     <MenuItem value="tcaPlus">TCA_PLUS</MenuItem>
//                     <MenuItem value="jda">JDA</MenuItem>
//                   </Select>
//                 </FormControl>
//               </Box>

//               <Box mt={2} mb={2} display="flex" justifyContent="center">
//                 <Button variant="contained" color="primary" onClick={handleSubmit} size="large" startIcon={<ModelTrainingIcon />} style={{ transition: 'all 0.3s', padding: '10px 30px' }}>
//                   Submit
//                 </Button>
//               </Box>

//               {loading && (
//                 <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
//                   <CircularProgress />
//                 </Box>
//               )}

//               {/* Results Section */}
//               {!loading && metrics && (
//                 <Box mt={4}>
//                   <Typography variant="h5" gutterBottom>
//                     General Metrics:
//                   </Typography>
//                   <Grid container spacing={3}>
//                     <Grid item xs={4}>
//                       <Paper elevation={3} style={{ padding: '20px', textAlign: 'center', boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)' }}>
//                         <CheckCircleIcon fontSize="large" color="success" />
//                         <Typography variant="h6" gutterBottom>
//                           Accuracy
//                         </Typography>
//                         <Typography variant="body1">{metrics.accuracy}</Typography>
//                       </Paper>
//                     </Grid>
//                     <Grid item xs={4}>
//                       <Paper elevation={3} style={{ padding: '20px', textAlign: 'center', boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)' }}>
//                         <AssessmentIcon fontSize="large" color="primary" />
//                         <Typography variant="h6" gutterBottom>
//                           AUC-ROC
//                         </Typography>
//                         <Typography variant="body1">{metrics.aucRoc}</Typography>
//                       </Paper>
//                     </Grid>
//                     <Grid item xs={4}>
//                       <Paper elevation={3} style={{ padding: '20px', textAlign: 'center', boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)' }}>
//                         <BarChartIcon fontSize="large" color="secondary" />
//                         <Typography variant="h6" gutterBottom>
//                           F1-Score
//                         </Typography>
//                         <Typography variant="body1">{metrics.f1Score}</Typography>
//                       </Paper>
//                     </Grid>
//                   </Grid>
//                 </Box>
//               )}

//               {!loading && targetData.length > 0 && (
//                 <Box mt={4}>
//                   <Typography variant="h5" gutterBottom>
//                     Target Data with Labels
//                   </Typography>
//                   <Table>
//                     <TableHead>
//                       <TableRow>
//                         {Object.keys(targetData[0]).map((key) => (
//                           <TableCell key={key}>{key}</TableCell>
//                         ))}
//                       </TableRow>
//                     </TableHead>
//                     <TableBody>
//                       {targetData.map((row, index) => (
//                         <TableRow key={index}>
//                           {Object.values(row).map((value, i) => (
//                             <TableCell key={i}>{value}</TableCell>
//                           ))}
//                         </TableRow>
//                       ))}
//                     </TableBody>
//                   </Table>
//                 </Box>
//               )}
//             </CardContent>
//           </Card>
//         </Fade>
//       </Container>
//     </ThemeProvider>
//   );
// }

// export default App;



//  kala
// 






//  new kala kala kala 
// import React, { useState } from 'react';
// import axios from 'axios';
// import {
//   Container,
//   Typography,
//   Button,
//   Input,
//   Card,
//   CardContent,
//   Table,
//   TableBody,
//   TableCell,
//   TableHead,
//   TableRow,
//   Box,
//   Grid,
//   Paper,
//   Select,
//   MenuItem,
//   FormControl,
//   InputLabel,
//   CircularProgress,
//   Grow,
//   Fade,
// } from '@mui/material';
// import CheckCircleIcon from '@mui/icons-material/CheckCircle';
// import AssessmentIcon from '@mui/icons-material/Assessment';
// import BarChartIcon from '@mui/icons-material/BarChart';
// import FileUploadIcon from '@mui/icons-material/FileUpload';
// import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
// import { createTheme, ThemeProvider } from '@mui/material/styles';

// const theme = createTheme({
//   palette: {
//     primary: {
//       main: '#673ab7',
//     },
//     secondary: {
//       main: '#ff9800',
//     },
//     success: {
//       main: '#4caf50',
//     },
//     error: {
//       main: '#f44336',
//     },
//   },
//   typography: {
//     fontFamily: 'Roboto, Arial, sans-serif',
//   },
// });

// function App() {
//   const [sourceFile, setSourceFile] = useState(null);
//   const [targetFile, setTargetFile] = useState(null);
//   const [metrics, setMetrics] = useState(null);
//   const [targetData, setTargetData] = useState([]);
//   const [selectedModel, setSelectedModel] = useState('pls');
//   const [loading, setLoading] = useState(false);

//   const handleSourceFileChange = (e) => {
//     setSourceFile(e.target.files[0]);
//   };

//   const handleTargetFileChange = (e) => {
//     setTargetFile(e.target.files[0]);
//   };

//   const handleModelChange = (e) => {
//     setSelectedModel(e.target.value);
//   };

//   const handleSubmit = async () => {
//     setLoading(true);
//     const formData = new FormData();
//     formData.append('source', sourceFile);
//     formData.append('target', targetFile);
//     formData.append('model_type', selectedModel);

//     try {
//       const response = await axios.post('http://localhost:5000/upload', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });
//       setMetrics(response.data);
//       setTargetData(response.data.targetDataWithLabels);
//     } catch (error) {
//       console.error('Error uploading files:', error);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <ThemeProvider theme={theme}>
//       {/* Full Page Background */}
//       <Box
//         style={{
//           minHeight: '100vh',
//           background: 'linear-gradient(to right, #6a11cb, #2575fc)', // Attractive gradient background
//           display: 'flex',
//           alignItems: 'center',
//           justifyContent: 'center',
//           padding: '20px',
//         }}
//       >
//         <Container maxWidth="md">
//           {/* Title Section */}
//           <Grow in={true} timeout={1000}>
//             <Card
//               variant="outlined"
//               style={{
//                 padding: '30px',
//                 background: '#ffffffee',
//                 boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)',
//               }}
//             >
//               <CardContent style={{ textAlign: 'center' }}>
//                 <Typography
//                   variant="h3"
//                   color="primary"
//                   style={{ fontWeight: 600, marginBottom: '20px' }}
//                 >
//                   Transfer Learning Tool
//                 </Typography>
//                 <Typography variant="subtitle1" color="textSecondary">
//                   Easily upload your source and target datasets, choose your preferred model, and let the tool do the magic!
//                 </Typography>
//               </CardContent>
//             </Card>
//           </Grow>

//           {/* Upload Section */}
//           <Fade in={true} timeout={1000}>
//             <Card
//               variant="outlined"
//               style={{
//                 marginTop: '40px',
//                 padding: '20px',
//                 background: '#ffffffee',
//                 boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
//               }}
//             >
//               <CardContent>
//                 <Typography variant="h4" color="secondary" gutterBottom>
//                   Upload Source and Target Files
//                 </Typography>
//                 <Box display="flex" justifyContent="center" alignItems="center" mb={2}>
//                   <FileUploadIcon fontSize="large" color="primary" />
//                 </Box>

//                 <Box mb={2}>
//                   <Input
//                     type="file"
//                     onChange={handleSourceFileChange}
//                     fullWidth={true}
//                     inputProps={{ style: { padding: '10px' } }}
//                   />
//                 </Box>
//                 <Box mb={2}>
//                   <Input
//                     type="file"
//                     onChange={handleTargetFileChange}
//                     fullWidth={true}
//                     inputProps={{ style: { padding: '10px' } }}
//                   />
//                 </Box>

//                 <Box mb={2}>
//                   <FormControl fullWidth>
//                     <InputLabel>Choose Method</InputLabel>
//                     <Select
//                       value={selectedModel}
//                       onChange={handleModelChange}
//                       label="Choose Method"
//                       style={{ padding: '10px' }}
//                     >
//                       <MenuItem value="pls">PLS</MenuItem>
//                       <MenuItem value="dpls">DPLS</MenuItem>
//                       <MenuItem value="gdpls">GDPLS</MenuItem>
//                       <MenuItem value="coral">CORAL</MenuItem>
//                       <MenuItem value="tca">TCA</MenuItem>
//                       <MenuItem value="tcaPlus">TCA_PLUS</MenuItem>
//                       <MenuItem value="jda">JDA</MenuItem>
//                       <MenuItem value="bda">BDA</MenuItem>
//                     </Select>
//                   </FormControl>
//                 </Box>

//                 <Box mt={2} mb={2} display="flex" justifyContent="center">
//                   <Button
//                     variant="contained"
//                     color="primary"
//                     onClick={handleSubmit}
//                     size="large"
//                     startIcon={<ModelTrainingIcon />}
//                     style={{ transition: 'all 0.3s', padding: '10px 30px' }}
//                   >
//                     Submit
//                   </Button>
//                 </Box>

//                 {loading && (
//                   <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
//                     <CircularProgress />
//                   </Box>
//                 )}

//                 {/* Results Section */}
//                 {!loading && metrics && (
//                   <Box mt={4}>
//                     <Typography variant="h5" gutterBottom>
//                       General Metrics:
//                     </Typography>
//                     <Grid container spacing={3}>
//                       <Grid item xs={4}>
//                         <Paper
//                           elevation={3}
//                           style={{
//                             padding: '20px',
//                             textAlign: 'center',
//                             boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)',
//                           }}
//                         >
//                           <CheckCircleIcon fontSize="large" color="success" />
//                           <Typography variant="h6" gutterBottom>
//                             Accuracy
//                           </Typography>
//                           <Typography variant="body1">{metrics.accuracy}</Typography>
//                         </Paper>
//                       </Grid>
//                       <Grid item xs={4}>
//                         <Paper
//                           elevation={3}
//                           style={{
//                             padding: '20px',
//                             textAlign: 'center',
//                             boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)',
//                           }}
//                         >
//                           <AssessmentIcon fontSize="large" color="primary" />
//                           <Typography variant="h6" gutterBottom>
//                             AUC-ROC
//                           </Typography>
//                           <Typography variant="body1">{metrics.aucRoc}</Typography>
//                         </Paper>
//                       </Grid>
//                       <Grid item xs={4}>
//                         <Paper
//                           elevation={3}
//                           style={{
//                             padding: '20px',
//                             textAlign: 'center',
//                             boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)',
//                           }}
//                         >
//                           <BarChartIcon fontSize="large" color="secondary" />
//                           <Typography variant="h6" gutterBottom>
//                             F1-Score
//                           </Typography>
//                           <Typography variant="body1">{metrics.f1Score}</Typography>
//                         </Paper>
//                       </Grid>
//                     </Grid>
//                   </Box>
//                 )}

//                 {!loading && targetData.length > 0 && (
//                   <Box mt={4}>
//                     <Typography variant="h5" gutterBottom>
//                       Target Data with Labels
//                     </Typography>
//                     <Box style={{ overflowX: 'auto' }}> {/* Allow horizontal scrolling */}
//                       <Table style={{ minWidth: '800px' }}> {/* Set a minimum width for the table */}
//                         <TableHead>
//                           <TableRow>
//                             {Object.keys(targetData[0]).map((key) => (
//                               <TableCell key={key}>{key}</TableCell>
//                             ))}
//                           </TableRow>
//                         </TableHead>
//                         <TableBody>
//                           {targetData.map((row, index) => (
//                             <TableRow key={index}>
//                               {Object.values(row).map((value, i) => (
//                                 <TableCell key={i}>{value}</TableCell>
//                               ))}
//                             </TableRow>
//                           ))}
//                         </TableBody>
//                       </Table>
//                     </Box>
//                   </Box>
//                 )}
//               </CardContent>
//             </Card>
//           </Fade>
//         </Container>
//       </Box>
//     </ThemeProvider>
//   );
// }

// export default App;




//download
import React, { useState } from 'react';
import axios from 'axios';
import {
  Container,
  Typography,
  Button,
  Input,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Box,
  Grid,
  Paper,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Grow,
  Fade,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import AssessmentIcon from '@mui/icons-material/Assessment';
import BarChartIcon from '@mui/icons-material/BarChart';
import FileUploadIcon from '@mui/icons-material/FileUpload';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';
import DownloadIcon from '@mui/icons-material/Download';
import { createTheme, ThemeProvider } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#673ab7',
    },
    secondary: {
      main: '#ff9800',
    },
    success: {
      main: '#4caf50',
    },
    error: {
      main: '#f44336',
    },
  },
  typography: {
    fontFamily: 'Roboto, Arial, sans-serif',
  },
});

function App() {
  const [sourceFile, setSourceFile] = useState(null);
  const [targetFile, setTargetFile] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [targetData, setTargetData] = useState([]);
  const [selectedModel, setSelectedModel] = useState('pls');
  const [loading, setLoading] = useState(false);

  const handleSourceFileChange = (e) => {
    setSourceFile(e.target.files[0]);
  };

  const handleTargetFileChange = (e) => {
    setTargetFile(e.target.files[0]);
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };

  const handleSubmit = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append('source', sourceFile);
    formData.append('target', targetFile);
    formData.append('model_type', selectedModel);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMetrics(response.data);
      setTargetData(response.data.targetDataWithLabels);
    } catch (error) {
      console.error('Error uploading files:', error);
    } finally {
      setLoading(false);
    }
  };

  // Function to convert targetData to CSV format
  const convertToCSV = (data) => {
    const headers = Object.keys(data[0]);
    const rows = data.map(row =>
      headers.map(header => row[header]).join(',')
    );
    return [headers.join(','), ...rows].join('\n');
  };

  // Function to download CSV file
  const downloadCSV = () => {
    const csv = convertToCSV(targetData);
    const blob = new Blob([csv], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'targetData.csv';
    link.click();
  };

  return (
    <ThemeProvider theme={theme}>
      {/* Full Page Background */}
      <Box
        style={{
          minHeight: '100vh',
          background: 'linear-gradient(to right, #6a11cb, #2575fc)', // Attractive gradient background
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '20px',
        }}
      >
        <Container maxWidth="md">
          {/* Title Section */}
          <Grow in={true} timeout={1000}>
            <Card
              variant="outlined"
              style={{
                padding: '30px',
                background: '#ffffffee',
                boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)',
              }}
            >
              <CardContent style={{ textAlign: 'center' }}>
                <Typography
                  variant="h3"
                  color="primary"
                  style={{ fontWeight: 600, marginBottom: '20px' }}
                >
                  Transfer Learning Tool
                </Typography>
                <Typography variant="subtitle1" color="textSecondary">
                  Easily upload your source and target datasets, choose your preferred model, and let the tool do the magic!
                </Typography>
              </CardContent>
            </Card>
          </Grow>

          {/* Upload Section */}
          <Fade in={true} timeout={1000}>
            <Card
              variant="outlined"
              style={{
                marginTop: '40px',
                padding: '20px',
                background: '#ffffffee',
                boxShadow: '0px 4px 20px rgba(0, 0, 0, 0.1)',
              }}
            >
              <CardContent>
                <Typography variant="h4" color="secondary" gutterBottom>
                  Upload Source and Target Files
                </Typography>
                <Box display="flex" justifyContent="center" alignItems="center" mb={2}>
                  <FileUploadIcon fontSize="large" color="primary" />
                </Box>

                {/* <Box mb={2}>
                  <Input
                    type="file"
                    onChange={handleSourceFileChange}
                    fullWidth={true}
                    inputProps={{ style: { padding: '10px' } }}
                  />
                </Box>
                <Box mb={2}>
                  <Input
                    type="file"
                    onChange={handleTargetFileChange}
                    fullWidth={true}
                    inputProps={{ style: { padding: '10px' } }}
                  />
                </Box> */}


                <Box mb={2}>
                  <Input
                    type="file"
                    onChange={handleSourceFileChange}
                    fullWidth={true}
                    inputProps={{ accept: '.csv, .xlsx, .xls', style: { padding: '10px' } }}
                  />
                  <Typography variant="caption" color="textSecondary">
                    Supported formats: CSV, XLSX, XLS
                  </Typography>
                </Box>
                <Box mb={2}>
                  <Input
                    type="file"
                    onChange={handleTargetFileChange}
                    fullWidth={true}
                    inputProps={{ accept: '.csv, .xlsx, .xls', style: { padding: '10px' } }}
                  />
                  <Typography variant="caption" color="textSecondary">
                    Supported formats: CSV, XLSX, XLS
                  </Typography>
                </Box>


                <Box mb={2}>
                  <FormControl fullWidth>
                    <InputLabel>Choose Method</InputLabel>
                    <Select
                      value={selectedModel}
                      onChange={handleModelChange}
                      label="Choose Method"
                      style={{ padding: '10px' }}
                    >
                      <MenuItem value="pls">PLS</MenuItem>
                      <MenuItem value="dpls">DPLS</MenuItem>
                      <MenuItem value="gdpls">GDPLS</MenuItem>
                      <MenuItem value="coral">CORAL</MenuItem>
                      <MenuItem value="deepCoral">DEEP_CORAL</MenuItem>
                      <MenuItem value="tca">TCA</MenuItem>
                      <MenuItem value="tcaPlus">TCA_PLUS</MenuItem>
                      <MenuItem value="jda">JDA</MenuItem>
                      <MenuItem value="bda">BDA</MenuItem>
                    </Select>
                  </FormControl>
                </Box>

                <Box mt={2} mb={2} display="flex" justifyContent="center">
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSubmit}
                    size="large"
                    startIcon={<ModelTrainingIcon />}
                    style={{ transition: 'all 0.3s', padding: '10px 30px' }}
                  >
                    Submit
                  </Button>
                </Box>

                {loading && (
                  <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
                    <CircularProgress />
                  </Box>
                )}

                {/* Results Section */}
                {!loading && metrics && (
                  <Box mt={4}>
                    <Typography variant="h5" gutterBottom>
                      General Metrics:
                    </Typography>
                    <Grid container spacing={3}>
                      <Grid item xs={4}>
                        <Paper
                          elevation={3}
                          style={{
                            padding: '20px',
                            textAlign: 'center',
                            boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)',
                          }}
                        >
                          <CheckCircleIcon fontSize="large" color="success" />
                          <Typography variant="h6" gutterBottom>
                            Accuracy
                          </Typography>
                          <Typography variant="body1">{metrics.accuracy}</Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={4}>
                        <Paper
                          elevation={3}
                          style={{
                            padding: '20px',
                            textAlign: 'center',
                            boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)',
                          }}
                        >
                          <AssessmentIcon fontSize="large" color="primary" />
                          <Typography variant="h6" gutterBottom>
                            AUC-ROC
                          </Typography>
                          <Typography variant="body1">{metrics.aucRoc}</Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={4}>
                        <Paper
                          elevation={3}
                          style={{
                            padding: '20px',
                            textAlign: 'center',
                            boxShadow: '0px 4px 15px rgba(0, 0, 0, 0.1)',
                          }}
                        >
                          <BarChartIcon fontSize="large" color="secondary" />
                          <Typography variant="h6" gutterBottom>
                            F1-Score
                          </Typography>
                          <Typography variant="body1">{metrics.f1Score}</Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  </Box>
                )}

                {!loading && targetData.length > 0 && (
                  <Box mt={4}>
                    <Typography variant="h5" gutterBottom>
                      Target Data with Labels
                    </Typography>
                    <Box style={{ overflowX: 'auto' }}> {/* Allow horizontal scrolling */}
                      <Table style={{ minWidth: '800px' }}> {/* Set a minimum width for the table */}
                        <TableHead>
                          <TableRow>
                            {Object.keys(targetData[0]).map((key) => (
                              <TableCell key={key}>{key}</TableCell>
                            ))}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {targetData.map((row, index) => (
                            <TableRow key={index}>
                              {Object.values(row).map((value, i) => (
                                <TableCell key={i}>{value}</TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </Box>

                    {/* Download Button */}
                    <Box mt={2} display="flex" justifyContent="center">
                      <Button
                        variant="contained"
                        color="secondary"
                        onClick={downloadCSV}
                        startIcon={<DownloadIcon />}
                      >
                        Download CSV
                      </Button>
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Fade>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
