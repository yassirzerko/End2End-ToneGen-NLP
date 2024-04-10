import './App.css';
import { useState, useEffect } from 'react';
import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link'
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import { Select, MenuItem, FormControl, InputLabel, Tooltip, Button, TextField } from '@mui/material';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import IconButton from '@mui/material/IconButton';
import InfoIcon from '@mui/icons-material/Info';
import Stack from '@mui/material/Stack';
import LinearProgress from '@mui/material/LinearProgress';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';
import {ReactTyped} from "react-typed";



const appBarComponent = () => {
  return (
    <AppBar position="static" sx={{ backgroundColor: '#333', height: '60px' }}>
    <Toolbar>
      <Typography variant="h5" component="div" sx={{ flexGrow: 1, color: '#fff', textAlign: 'center'}}>
      Text tone classifier
      </Typography>
      <Link href="https://github.com/yassirzerko/End2End-ToneGen-NLP" color="inherit" underline="none" sx={{ mr: 2 }}>
        <GitHubIcon fontSize='large'/>
      </Link>
      <Link href="https://www.linkedin.com/in/yassir-el-aoufir/" color="inherit" underline="none">
        <LinkedInIcon fontSize='large'/>
      </Link>
    </Toolbar>
  </AppBar>
  )
}


const errorForm = (message) => {
  return (
    <div style={{ marginTop: '20px', textAlign: 'center' }}>
      <Alert severity="error" sx={{ width: '100%', maxWidth: '600px', margin: '0 auto' }}>
        <AlertTitle>Error</AlertTitle>
        {message}
      </Alert>
    </div>
  )
}

const customToolTip = (text, openToolTip) => {
  return (
    <Tooltip title= {text}>
      <IconButton onClick={openToolTip}>
        <InfoIcon />
      </IconButton>
    </Tooltip>
  );
}

const selectComponenent = (labelName, selectedItem, data, handleChange, disabled, tooltipText, openToolTip) => {

  return (
      <FormControl variant="outlined"  sx={{marginTop : 2, marginBottom:2}}>
        <InputLabel id="select-label" sx = {{marginLeft : -1.5, marginTop : 0}}>{labelName}</InputLabel>
        <Stack direction={'row'} > 
        <Select
          labelId="select-label"
          id="select"
          value={selectedItem}
          onChange={handleChange}
          disabled={disabled}
          sx={{minWidth:300, marginTop: 1}}
        >
          {data.map((option, index) => (
            <MenuItem key={index} value={option}>
              {option}
            </MenuItem>
          ))}
        </Select>
        {customToolTip(tooltipText, openToolTip)}
        </Stack>
        
      </FormControl>
  );
}


// API DATA : Dict : with two keys : Models, Feature vector format
// Each value is a dict : for the first : model_id : [name, performance, description,]
const API = 'api_test'
const models_data = [{id : 'id1', model_name: 'rf', bow_acc : 0.90,tf_idf_acc : 0.94, w2v_max_acc : 0.93, w2v_sum_acc : 0.40,  w2v_sum_mean : 0.60,},
{id : 'id2', model_name: 'gb', bow_acc : 0.90,tf_idf_acc : 0.94, w2v_max_acc : 0.93, w2v_sum_acc : 0.40,  w2v_sum_mean : 0.60,},
{id : 'id3', model_name: 'mlp', bow_acc : 0.90,tf_idf_acc : 0.94, w2v_max_acc : 0.93, w2v_sum_acc : 0.40,  w2v_sum_mean : 0.60,},
{id : 'id4', model_name: 'ada', bow_acc : 0.90,tf_idf_acc : 0.94, w2v_max_acc : 0.93, w2v_sum_acc : 0.40,  w2v_sum_mean : 0.60,},]

const Mock_data = {models : models_data , feature_format : ['bow', 'tf_idf', 'w2v_max', 'w2v_sum', 'w2v_mean']}
function App() {
  const [inputText,setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [choosenModel, setChoosenModel] = useState('')
  const [featureFormat, setFeatureFormat] = useState('')
  const [modelsData, setModelsData] = useState([])
  const [featuresData, setFeaturesData] = useState([])
  const [error, setError] = useState('')
  const [openModal, setOpenModal] = useState(false)
  const [modalText, setModalText] = useState(['',''])

  const fetchApiData = async () => {
    try {
      setIsLoading(true)
      const response = await fetch(API)
      if (!response.ok) {
        setError(response)
        setIsLoading(false)
        
        return
      }
      setIsLoading(true)
      const data = await response.json()
      setIsLoading(false)
      //setApiData(data)

    }
    catch(error) {
      console.log(error)
      setError(error.message)
      setIsLoading(false)
      setModelsData(Mock_data['models'])
      setFeaturesData(Mock_data['feature_format'])
      setChoosenModel(Mock_data['models'][0]['model_name'])
      setFeatureFormat(Mock_data['feature_format'][0])

    }
  }

  const submitText = async () => {
    // first check if the text is correct
    try {
      setIsLoading(true)
      const response = await fetch(API + inputText)
      if (!response.ok) {
        setError(response)
        setIsLoading(false)
        return
      }
      setIsLoading(true)
      const data = await response.json()
      setIsLoading(false)
      //setPrediction(data)

    }
    catch(error) {
      console.log(error)
      //setError(error.message)
      setTimeout(() => {
        console.log("Delayed for 1 second.");
        setIsLoading(false)
        setModalText(['Model prediction', [['','Your text was labeled as having a ' + 0 + ' tone']]])
        setOpenModal(true)
      }, "1000");
      
    }
    
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




  const get_accuracy_sentence = (modelsData, modelName, featureFormat) => {
    for (let idx in modelsData) {
      
      if (modelsData[idx]['model_name'] == modelName) {
        return `With the choosen configuration, with the model ${modelName} and the feature representation ${featureFormat}, the model had an accuracy of ${modelsData[idx][featureFormat+'_acc']} on the test set.`
      }
    } 
  }


  // Component will mount
  useEffect(() => {fetchApiData()}, [])


  //const disabled = isLoading || error !== ''
  const disabled = isLoading
  const modelsNames = modelsData.map(model => model.model_name)
  const appExplanation = `
Welcome to our Tone Prediction Application! 
Choose a trained model and feature representation technique from the dropdown menus, then input your text to predict its tone.
Tones Available for Prediction:
Formal,
Casual,
Serious,
Humorous,
Professional,
Instructive,
Persuasive,
Narrative,
Analytical,
Empathetic.
Get started now and discover the tone of your text!
`;

const featureRepresentations = [
  ["General explanation", "Feature representation techniques transform text into a format that machine learning models can understand. In this project, five distinct methods of feature representation are employed to characterize texts."],
  ["Bag of Visual Words (BoVW)", "Represent text by counting occurrences of visual word 'clusters' in a predefined vocabulary."],
  ["Term Frequency-Inverse Document Frequency (TF-IDF)", "Weigh each word by its frequency in the document and its rarity across documents."],
  ["Sum of Each Dimension", "Calculate the sum of word embeddings across each dimension in the Word2Vec space."],
  ["Mean of Each Dimension", "Compute the average of word embeddings across each dimension in the Word2Vec space."],
  ["Max of Each Dimension", "Determine the maximum value of word embeddings across each dimension in the Word2Vec space."]
];

const modelDescriptions = [
  ["General explanation", "Various machine learning models are employed to predict the tone of input texts. In this project, a selection of models is utilized, each offering unique approaches to tone classification."],
  ["Random Forest", "A versatile ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification problems."],
  ["Gradient Boosting", "An ensemble learning technique that builds a strong model by iteratively adding weak learners (decision trees) to minimize the loss function."],
  ["Histogram Gradient Boosting", "Similar to Gradient Boosting, this method builds an ensemble of decision trees, but uses histogram-based techniques for faster training and better accuracy."],
  ["AdaBoosting", "A boosting algorithm that combines multiple weak classifiers to form a strong classifier by focusing on the mistakes of the previous classifiers."],
  ["Neural Network", "A deep learning model inspired by the structure of the human brain, consisting of interconnected nodes organized in layers. It's capable of learning complex patterns in data."],
];

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
          {selectComponenent('Choose a feature representation', featureFormat, featuresData, handleFeatureChange, disabled, 'Learn more',() => {
            setModalText(['Feature Representation Techniques', featureRepresentations])
            setOpenModal(true)
          })}
          
          
          </Box>
          
          {choosenModel !== null && 
          <Typography variant="h6"  gutterBottom sx={{ fontFamily: 'Calibri', textAlign: 'center', marginTop : 3 }}>
          {get_accuracy_sentence(modelsData, choosenModel, featureFormat)}
        </Typography>}
          <Box sx={{ marginTop: 5 }}>
                <TextField
                  disabled={disabled}
                  onChange={handleFormChange}
                  label="Input the text for which you'd like to predict the tone"
                  fullWidth
                  multiline
                  variant="outlined"
                  rows={2}
                />
                <Button size='large' onClick={submitText} disabled={disabled} variant="contained" sx={{ marginTop: 5, marginLeft:'40%'}}>
                  Predict text tone
                </Button>
          
            </Box>
        </CardContent>
        </Card>
      
      }
      {openModal && 
      <Dialog
      open={openModal}
      onClose={() => setOpenModal(false)}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle id="alert-dialog-title" sx={{marginLeft : 0}}>
        {modalText[0]}
      </DialogTitle>

      {modalText[1].map( ([feature_name, description],idx) => 
        <DialogContent>
        <Typography>{feature_name}</Typography>
          <DialogContentText id="alert-dialog-description">
            <Typography>{description}</Typography>
            
          </DialogContentText>
        </DialogContent>
      )}
    </Dialog>}
      {isLoading && <LinearProgress sx={{width : '50%', marginLeft : '25%', marginRight : '25%', marginTop: 3}}></LinearProgress>}
    </Box>
    
  );
}

export default App;
