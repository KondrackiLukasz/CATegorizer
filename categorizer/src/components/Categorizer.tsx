import React, { ChangeEvent, useState } from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Container, Grid, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow } from '@mui/material';
import { Pets } from '@mui/icons-material';
import styled from '@emotion/styled';

const Input = styled('input')({
    display: 'none',
});

const mockData = [
    { name: 'Russian blue [K]', probability: 0.92 },
    { name: 'British shorthair [K]', probability: 0.04 },
    { name: 'Havanese [P]', probability: 0.01 },
    { name: 'Ragdoll [K]', probability: 0.01 },
    { name: 'Samoyed [P]', probability: 0.01 },
    { name: 'Shiba inu [P]', probability: 0.01 },
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
                setImageSrc(e.target?.result as string);
            };
            reader.readAsDataURL(file);
        }
    };

    const handleClearImage = () => {
        setImageSrc(null);
    };

    return (
        <div>
            <AppBar position="static" sx={{ borderBottom: 'none', boxShadow: 'none' }}>
                <Toolbar>
                    <Pets sx={{ marginRight: 1 }} />
                    <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
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
            <Container maxWidth="lg" sx={{ marginTop: '1rem', flexGrow: 1 }}>
                <Grid container spacing={3} sx={{ height: 'calc(100vh - 64px)' }}>
                    {imageSrc &&
                        pictureLabels.map((label, index) => (
                            <Grid key={index} item xs={12} sm={6} md={4} sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                                <Paper sx={{ padding: 1, textAlign: 'center', height: '100%', flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                                    <img
                                        src={imageSrc}
                                        alt={label}
                                        style={{ maxWidth: '100%', maxHeight: '80%', objectFit: 'contain' }}
                                    />
                                    <Typography variant="h6" sx={{ marginTop: 1 }}>{label}</Typography>
                                </Paper>
                            </Grid>
                        ))}
                </Grid>
                {imageSrc && (
                    <TableContainer component={Paper} sx={{ marginTop: 2 }}>
                        <Table sx={{minWidth: 650 }} aria-label="mock data table">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Name</TableCell>
                                    <TableCell align="right">Probability</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {mockData.map((row, index) => (
                                    <TableRow key={index}>
                                        <TableCell component="th" scope="row">
                                            {row.name}
                                        </TableCell>
                                        <TableCell align="right">{row.probability}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </TableContainer>
                )}
            </Container>
        </div>
    );
}
