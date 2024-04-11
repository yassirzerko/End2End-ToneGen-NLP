import './App.css';
import { useState, useEffect } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import {Button, TextField } from '@mui/material';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import LinearProgress from '@mui/material/LinearProgress';
import {ReactTyped} from "react-typed";
import {featureRepresentations, modelDescriptions, server_URI, appExplanation} from './constants.js';
import {get_accuracy_sentence, buildRequest} from './utils.js'
import { modalComponent , appBarComponent, errorForm, selectComponenent} from './components.js';

function App() {
  const [inputText,setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [choosenModel, setChoosenModel] = useState(modelDescriptions[1][0])
  const [featureFormat, setFeatureFormat] = useState(featureRepresentations[1][0])
  const [modelsData, setModelsData] = useState([])
  const [error, setError] = useState('')
  const [openModal, setOpenModal] = useState(false)
  const [modalText, setModalText] = useState(['',''])
  
  const fetchApiData = async () => {
    setIsLoading(true)
    try {
      const response = await fetch(server_URI)
      if (!response.ok) {
        setError(response) 
      }
      else
      {
        const data = await response.json()
        setModelsData(data)
      }
      
    }
    catch(error) {
      console.log(error)
      setError(error.message)
    }
    setIsLoading(false)
  }

  const submitText = async () => {
    // first check if the text is correct
    setIsLoading(true)
    try {
      // also use choosen model and choosen encoding
      const response = await fetch(buildRequest(choosenModel, featureFormat, inputText))
      if (!response.ok) {
        setError(response)
      }
      else{
        const data = await response.json()
        setModalText(['Model prediction', [['','Your text was labeled as having a ' + data['prediction']+ ' tone']]])
        setOpenModal(true)
      }
       
    }
    catch(error) {
      setError(error.message)
      
    }
    setIsLoading(false)
    
  }

const handleModelChange = (event) => {
  setChoosenModel(event.target.value)
}

const handleFeatureChange = (event) => {
  setFeatureFormat(event.target.value)
}

const handleFormChange = (event) => {
  setInputText(event.target.value)
}

  // Component will mount
  useEffect(() => {fetchApiData()}, [])


  const disabled = isLoading || error !== ''
  const modelsNames = modelDescriptions.slice(1).map(model => model[0])
  const featureFormatNames = featureRepresentations.slice(1).map(feature => feature[0])
  const submit_disbaled =  inputText.split(/\s+/).length < 5
  
  return (
    <Box sx={{ flexGrow: 1 }}>
      {appBarComponent()}
      <Typography variant="h5" component="div"  sx={{width : '50%', marginLeft : '25%', marginRight : '25%', marginTop: 3}}> 
      <ReactTyped strings={[appExplanation]} typeSpeed={40}></ReactTyped>
      </Typography>
      {error !== '' &&  errorForm(error)}
      {isLoading && <LinearProgress sx={{width : '50%', marginLeft : '25%', marginRight : '25%', marginTop: 3}}></LinearProgress>}
      {
        <Card variant='outlined' sx={{ backgroundColor: '#F5F5F5', width: '60%',
        marginLeft: '20%',
        marginRight: '20%',
        marginTop : '1%',
    }} >
        <CardContent sx={{  width: '94%',
        marginLeft: '3%',
        marginRight: '3%',
        marginTop : '1%',
    }}>
          <Box  sx={{ display: 'flex', flexDirection: 'column', marginLeft: '35%'}}>
          
          {selectComponenent('Choose a model', choosenModel, modelsNames, handleModelChange, disabled, 'Learn more', () => {
            setModalText(['Models', modelDescriptions])
            setOpenModal(true)
          })}
          {selectComponenent('Choose a feature representation', featureFormat, featureFormatNames, handleFeatureChange, disabled, 'Learn more',() => {
            setModalText(['Feature Representation Techniques', featureRepresentations])
            setOpenModal(true)
          })}
          
          
          </Box>
          
          {modelsData.length > 0 && 
          <Typography variant="h6"  gutterBottom sx={{ fontFamily: 'Calibri', textAlign: 'center', marginTop : 3 }}>
          {get_accuracy_sentence(modelsData, choosenModel, featureFormat)}
        </Typography>}

          <Box sx={{ marginTop: 5 }}>
                <TextField
                  disabled={disabled}
                  onChange={handleFormChange}
                  label="Input the text for which you'd like to predict the tone. Enter at least 5 words."
                  fullWidth
                  multiline
                  variant="outlined"
                  rows={2}
                />
                <Button size='large' onClick={submitText} disabled={disabled || submit_disbaled} variant="contained" sx={{ marginTop: 5, marginLeft:'40%'}}>
                  Predict text tone
                </Button>
          
            </Box>
        </CardContent>
        </Card>
      
      }
      {openModal && 
      modalComponent(openModal, ()=> setOpenModal(false), modalText[0], modalText[1])}
      {isLoading && <LinearProgress sx={{width : '50%', marginLeft : '25%', marginRight : '25%', marginTop: 3}}></LinearProgress>}
    </Box>
    
  );
}

export default App;
