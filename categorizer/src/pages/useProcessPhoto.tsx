import { useState, useEffect } from 'react';

type ProcessPhotoResult = {
    roi_adjusted_photo: string;
    classified_photo: string;
    classification_stats: Array<{
        breed: string;
        probability: number;
    }>;
};

export function useProcessPhoto(imageSrc: string | undefined) {
    const [processing, setProcessing] = useState(false);
    const [processedData, setProcessedData] = useState<ProcessPhotoResult | null>(null);

    useEffect(() => {
        if (!imageSrc) {
            setProcessedData(null);
            return;
        }

        const processPhoto = async () => {
            setProcessing(true);
            const base64Image = imageSrc.replace(/^data:image\/\w+;base64,/, '');

            try {
                const response = await fetch('http://localhost:8080/process_photo', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ photo: base64Image }),
                });

                if (response.ok) {
                    const data: ProcessPhotoResult = await response.json();
                    data.roi_adjusted_photo = `data:image/jpeg;base64,${data.roi_adjusted_photo}`;
                    data.classified_photo = `data:image/jpeg;base64,${data.classified_photo}`;
                    setProcessedData(data);
                } else {
                    console.error('Failed to process photo:', response.statusText);
                }
            } catch (error) {
                console.error('Error processing photo:', error);
            } finally {
                setProcessing(false);
            }
        };

        processPhoto();
    }, [imageSrc]);

    return { processing, processedData };
}
