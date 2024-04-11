import './App.css';
import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link'
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import { Select, MenuItem, FormControl, InputLabel, Tooltip } from '@mui/material';
import IconButton from '@mui/material/IconButton';
import InfoIcon from '@mui/icons-material/Info';
import Stack from '@mui/material/Stack';
import Dialog from '@mui/material/Dialog';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogTitle from '@mui/material/DialogTitle';


export const appBarComponent = () => {
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
  
  
export const errorForm = (message) => {
    return (
      <div style={{ marginTop: '20px', textAlign: 'center' }}>
        <Alert severity="error" sx={{ width: '100%', maxWidth: '600px', margin: '0 auto' }}>
          <AlertTitle>Error</AlertTitle>
          {message}
        </Alert>
      </div>
    )
  }
  
export const selectComponenent = (labelName, selectedItem, data, handleChange, disabled, tooltipText, openToolTip) => {
  
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
          <Tooltip title= {tooltipText}>
        <IconButton onClick={openToolTip}>
          <InfoIcon />
        </IconButton>
      </Tooltip>
          </Stack>
          
        </FormControl>
    );
  }

export const modalComponent = (openModal, onClose, modal_title, modal_paragraphs_data) => {
    return(
    <Dialog
      open={openModal}
      onClose={onClose}
      aria-labelledby="alert-dialog-title"
      aria-describedby="alert-dialog-description"
    >
      <DialogTitle id="alert-dialog-title" sx={{marginLeft : 0}}>
        {modal_title}
      </DialogTitle>

      {modal_paragraphs_data.map( ([title, text],idx) => 
        <DialogContent>
        <Typography>{title}</Typography>
          <DialogContentText id="alert-dialog-description">
            <Typography>{text}</Typography>
            
          </DialogContentText>
        </DialogContent>
      )}
    </Dialog>
    )
}
  