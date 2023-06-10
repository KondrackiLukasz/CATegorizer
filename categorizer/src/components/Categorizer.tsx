import React, {ChangeEvent, useState} from 'react';
import {
    AppBar,
    Toolbar,
    Typography,
    Button,
    Box,
    Container,
    Grid,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow
} from '@mui/material';
import {Pets} from '@mui/icons-material';
import styled from '@emotion/styled';
import {useProcessPhoto} from '@/pages/useProcessPhoto';

const Input = styled('input')({
    display: 'none',
});

const mockData = [
    {name: 'Russian blue [K]', probability: 0.92},
    {name: 'British shorthair [K]', probability: 0.04},
    {name: 'Havanese [P]', probability: 0.01},
    {name: 'Ragdoll [K]', probability: 0.01},
    {name: 'Samoyed [P]', probability: 0.01},
    {name: 'Shiba inu [P]', probability: 0.01},
];

const pictureLabels = [
    'Original photo',
    'Adjusted photo',
    'Final photo',
];

export default function Categorizer() {
    const [imageSrc, setImageSrc] = useState<string | null>(null);

    const handleImageUpload = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files && event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                if (typeof e.target?.result === 'string') {
                    setImageSrc(e.target.result);
                }
            };
            reader.readAsDataURL(file);
        }
    };

    const handleClearImage = () => {
        setImageSrc(null);
    };

    const {processing, processedData} = useProcessPhoto(imageSrc ?? undefined);

    return (
        <div>
            <AppBar position="static" sx={{borderBottom: 'none', boxShadow: 'none'}}>
                <Toolbar>
                    <Pets sx={{marginRight: 1}}/>
                    <Typography variant="h6" component="div" sx={{flexGrow: 1}}>
                        Categorizer
                    </Typography>
                    {imageSrc ? (
                        <Button variant="outlined" color="inherit" onClick={handleClearImage}>
                            Clear
                        </Button>
                    ) : (
                        <label htmlFor="image-upload">
                            <Input
                                accept="image/*"
                                id="image-upload"
                                type="file"
                                onChange={handleImageUpload}
                            />
                            <Button variant="outlined" color="inherit" component="span">
                                Upload Image
                            </Button>
                        </label>
                    )}
                </Toolbar>
            </AppBar>
            <Container maxWidth="lg" sx={{marginTop: '1rem',  flexGrow: 1}}>
                {processedData && (
                    <TableContainer component={Paper} sx={{marginTop: 2, marginBottom: 2}}>
                        <Table sx={{minWidth: 650}} aria-label="classification stats table">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Name</TableCell>
                                    <TableCell align="right">Probability</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {processedData.classification_stats.map((row, index) => (
                                    <TableRow key={index}>
                                        <TableCell component="th" scope="row">
                                            {row.breed}
                                        </TableCell>
                                        <TableCell align="right">{row.probability}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                )}
                <Grid container spacing={3} sx={{height: 'calc(100vh - 64px)', marginBottom: 1}}>
                    {processedData &&
                        [imageSrc, processedData.roi_adjusted_photo, processedData.classified_photo].map((src, index) => (
                            <Grid key={index} item xs={12} sm={6} md={4} sx={{
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'flex-start',
                                minHeight: 'auto'
                            }}>
                                <Paper
                                    sx={{
                                        padding: 1,

                                        textAlign: 'center',
                                        flexGrow: 1,
                                        display: 'flex',
                                        flexDirection: 'column',
                                        justifyContent: 'center',
                                        flexBasis: 'auto',
                                    }}
                                > <img
                                    src={src ?? undefined}
                                    alt={pictureLabels[index]}
                                    style={{maxWidth: '100%', maxHeight: '80%', objectFit: 'contain'}}
                                />
                                    <Typography variant="h6" sx={{marginTop: 1}}>{pictureLabels[index]}</Typography>
                                </Paper>
                            </Grid>
                        ))}
                </Grid>

            </Container>
        </div>
    );

}
